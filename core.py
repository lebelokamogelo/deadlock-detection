from __future__ import annotations
import time, random, threading, queue
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Deque, Optional, Set, List
import numpy as np
import networkx as nx

# ----------------------------- Simulation core -----------------------------
@dataclass
class LogRecord:
    level: str
    event: str
    tid: Optional[str] = None
    res: Optional[str] = None
    details: Optional[str] = None
    ts: float = field(default_factory=time.time)  # timestamp at creation

    def as_dict(self):
        return {
            "time": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts)),
            "level": self.level,
            "event": self.event,
            "tid": self.tid,
            "res": self.res,
            "details": self.details,
        }

@dataclass
class ResourceState:
    holder: Optional[str] = None
    wait_queue: Deque[str] = field(default_factory=deque)

class Logger:
    """
    Thread-safe logger with buffering and pause/resume.

    Maintains both a queue for real-time consumption and a circular buffer
    for historical log viewing. Supports pausing to freeze log state during
    deadlock analysis.
    """

    def __init__(self, max_buffer_size: Optional[int] = None):
        self.q: queue.Queue[LogRecord] = queue.Queue()
        self.buffer: List[LogRecord] = []
        self.lock = threading.Lock()
        self.max_buffer_size = max_buffer_size  # None = unbounded
        self.enabled = True  # pause/resume switch

    def pause(self):
        self.enabled = False

    def resume(self):
        self.enabled = True

    def log(self, level: str, event: str, tid: Optional[str] = None, res: Optional[str] = None,
            details: Optional[str] = None, *, force: bool = False):
        # When paused, ignore logs unless force=True (so RESOLUTION can still be recorded)
        if not self.enabled and not force:
            return
        rec = LogRecord(level, event, tid, res, details)
        self.q.put(rec)
        with self.lock:
            self.buffer.append(rec)
            if self.max_buffer_size and len(self.buffer) > self.max_buffer_size:
                self.buffer = self.buffer[-self.max_buffer_size:]

    def drain(self) -> List[LogRecord]:
        drained: List[LogRecord] = []
        while not self.q.empty():
            drained.append(self.q.get())
        return drained

class ResourceManager:
    """
    Thread-safe resource allocation manager with deadlock detection support.

    Manages a pool of named resources, tracking which threads hold which resources
    and maintaining wait queues. Builds wait-for graphs for deadlock detection
    and supports thread termination for deadlock resolution.
    """
    def __init__(self, resource_ids: List[str], logger: Logger):
        self.resources: Dict[str, ResourceState] = {r: ResourceState() for r in resource_ids}
        self.held_by_thread: Dict[str, Set[str]] = defaultdict(set)
        self.waiting_for: Dict[str, Optional[str]] = defaultdict(lambda: None)
        self.waiting_since: Dict[str, Optional[float]] = defaultdict(lambda: None)  # when a thread started waiting

        # Synchronization and state management
        self.cv = threading.Condition()
        self.logger = logger
        self.aborted: Set[str] = set()

    def try_acquire(self, r: str, tid: str) -> bool:
        with self.cv:
            stt = self.resources[r]
            if stt.holder is None:
                stt.holder = tid
                self.held_by_thread[tid].add(r)
                self.waiting_since[tid] = None
                self.logger.log("INFO", "GRANT", tid, r)
                return True
            # Re-entrant request
            if stt.holder == tid:
                self.logger.log("INFO", "REENTRANT", tid, r, details="already holds; treat as granted")
                return True
            # Otherwise, block
            if tid not in stt.wait_queue:
                stt.wait_queue.append(tid)
            self.waiting_for[tid] = r
            if self.waiting_since[tid] is None:
                self.waiting_since[tid] = time.time()
            self.logger.log("WARN", "BLOCK", tid, r, details=f"holder={stt.holder}")
            return False

    def acquire(self, resources: List[str], tid: str, shuffle=True):
        order = random.sample(resources, len(resources)) if shuffle else sorted(resources)
        for r in order:
            # Randomize order to reduce likelihood of circular wait conditions
            while True:
                if tid in self.aborted:
                    return False
                if self.try_acquire(r, tid):
                    break
                with self.cv:
                    self.cv.wait(timeout=0.2)
        return True

    def release_all(self, tid: str):
        with self.cv:
            held = list(self.held_by_thread.get(tid, set()))
            for r in held:
                stt = self.resources[r]
                if stt.holder == tid:
                    stt.holder = None
                    self.logger.log("INFO", "RELEASE", tid, r)
                    # Wake the first non-aborted waiter
                    if stt.wait_queue:
                        nxt = None
                        while stt.wait_queue and nxt is None:
                            cand = stt.wait_queue.popleft()
                            if cand not in self.aborted:
                                nxt = cand
                        if nxt:
                            stt.holder = nxt
                            self.held_by_thread[nxt].add(r)
                            self.waiting_for[nxt] = None
                            self.waiting_since[nxt] = None
                            self.logger.log("INFO", "GRANT", nxt, r, details="wakeup")
            self.held_by_thread[tid].clear()
            self.waiting_for[tid] = None
            self.waiting_since[tid] = None
            self.cv.notify_all()

    def build_wfg(self) -> nx.DiGraph:
        g = nx.DiGraph()
        with self.cv:
            for t in set(list(self.held_by_thread.keys()) + list(self.waiting_for.keys())):
                g.add_node(t)
            for r, stt in self.resources.items():
                if stt.holder is None:
                    continue
                holder = stt.holder
                for waiter in list(stt.wait_queue):
                    g.add_edge(waiter, holder, resource=r)
        return g

    def build_rag(self) -> nx.DiGraph:
        g = nx.DiGraph()
        with self.cv:
            for r in self.resources:
                g.add_node(r, kind="res")
            for t in set(list(self.held_by_thread.keys()) + list(self.waiting_for.keys())):
                g.add_node(t, kind="proc")
            for r, stt in self.resources.items():
                if stt.holder:
                    g.add_edge(stt.holder, r, kind="holds")
                for waiter in list(stt.wait_queue):
                    g.add_edge(waiter, r, kind="waits")
        return g

    def edge_resource(self, waiter: str, holder: str) -> Optional[str]:
        with self.cv:
            for r, stt in self.resources.items():
                if stt.holder == holder and waiter in stt.wait_queue:
                    return r
        return None

    def abort(self, tid: str):
        """
        Forcibly terminate a thread to resolve deadlocks.

        Removes the thread from all wait queues, releases all held resources,
        and marks it as aborted to prevent future resource requests.
        """
        with self.cv:
            self.aborted.add(tid)
            self.logger.log("ERROR", "TERMINATE", tid, details="victim termination", force=True)
            # Purge from all wait queues
            for stt in self.resources.values():
                try:
                    while tid in stt.wait_queue:
                        stt.wait_queue.remove(tid)
                except ValueError:
                    pass
            # Release any held resources
            self.release_all(tid)
            self.waiting_for[tid] = None
            self.waiting_since[tid] = None
            self.cv.notify_all()
