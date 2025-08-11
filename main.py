from __future__ import annotations
import time, random, threading, queue
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Deque, Optional, Set, List, Tuple
import numpy as np
import networkx as nx
import psutil
import streamlit as st
import matplotlib.pyplot as plt

# ----------------------------- Simulation core -----------------------------
@dataclass
class LogRecord:
    level: str
    event: str
    tid: Optional[str] = None
    res: Optional[str] = None
    details: Optional[str] = None
    def as_dict(self):
        return {
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
    def __init__(self, max_buffer_size: Optional[int] = None):
        self.q: queue.Queue[LogRecord] = queue.Queue()
        self.buffer: List[LogRecord] = []
        self.lock = threading.Lock()
        self.max_buffer_size = max_buffer_size  # None = unbounded

    def log(self, level: str, event: str, tid: Optional[str] = None, res: Optional[str] = None, details: Optional[str] = None):
        rec = LogRecord(level, event, tid, res, details)
        self.q.put(rec)
        with self.lock:
            self.buffer.append(rec)
            if self.max_buffer_size and len(self.buffer) > self.max_buffer_size:
                # only trim if a size is explicitly set
                self.buffer = self.buffer[-self.max_buffer_size:]

    def drain(self) -> List[LogRecord]:
        drained: List[LogRecord] = []
        while not self.q.empty():
            drained.append(self.q.get())
        return drained

class ResourceManager:
    def __init__(self, resource_ids: List[str], logger: Logger):
        self.resources: Dict[str, ResourceState] = {r: ResourceState() for r in resource_ids}
        self.held_by_thread: Dict[str, Set[str]] = defaultdict(set)
        self.waiting_for: Dict[str, Optional[str]] = defaultdict(lambda: None)
        self.waiting_since: Dict[str, Optional[float]] = defaultdict(lambda: None)  # when a thread started waiting
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
        with self.cv:
            self.aborted.add(tid)
            self.logger.log("ERROR", "TERMINATE", tid, details="victim termination")
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

class DeadlockDetector(threading.Thread):
    def __init__(self, rm: ResourceManager, logger: Logger, interval: float = 0.5, on_deadlock=None):
        super().__init__(daemon=True)
        self.rm = rm
        self.logger = logger
        self.interval = interval
        self.running = True
        self.on_deadlock = on_deadlock

    def run(self):
        while self.running:
            g = self.rm.build_wfg()
            try:
                cyc = nx.find_cycle(g, orientation="original")
                edges = [(u, v) for u, v, _ in cyc]
                nodes = list({u for u, v in edges} | {v for u, v in edges})
                self_waiters = [u for (u, v) in edges if u == v]
                deadlock_type = "self-wait" if self_waiters else "cycle"
                edge_details = [{"waiter": u, "holder": v, "resource": self.rm.edge_resource(u, v)} for (u, v) in edges]
                self.logger.log("ERROR", "DEADLOCK_DETECTED", details=f"type={deadlock_type} cycle={'→'.join(nodes)}")
                if self.on_deadlock:
                    self.on_deadlock({
                        "nodes": nodes,
                        "edges": edges,
                        "edge_details": edge_details,
                        "deadlock_type": deadlock_type,
                        "self_waiters": self_waiters
                    })
            except nx.exception.NetworkXNoCycle:
                pass
            time.sleep(self.interval)

    # -------- Victim selection with explanation --------
    @staticmethod
    def _features_for(rm: ResourceManager, tid: str) -> dict:
        # How many resources it holds
        holds = list(rm.held_by_thread.get(tid, set()))
        holds_count = len(holds)
        # How many threads are waiting on resources it holds (how much we unblock)
        dependents = 0
        with rm.cv:
            for r in holds:
                stt = rm.resources[r]
                dependents += len(stt.wait_queue)
        # How long this thread has been waiting (favor aborting newer waiters)
        ws = rm.waiting_since.get(tid)
        wait_age = 0.0 if ws is None else max(0.0, time.time() - ws)
        return {"holds_count": holds_count, "dependents": dependents, "wait_age": wait_age}

    @staticmethod
    def choose_victim_explained(rm: ResourceManager, candidates: List[str], policy: str = "min_cost") -> Tuple[str, dict]:
        """
        policy options:
          - 'min_cost'      : abort fewest-held-resources (cheap rollback), tie-break by lowest wait_age then dependents
          - 'max_unblock'   : abort the one that unblocks the most others (highest dependents), tie-break by holds then wait_age
          - 'most_resources': current classic heuristic (holds_count desc), tie-break by dependents desc, then wait_age
        Returns (victim_tid, explanation_dict)
        """
        feats = {t: DeadlockDetector._features_for(rm, t) for t in candidates}

        def key_min_cost(t):
            f = feats[t]
            return (f["holds_count"], f["wait_age"], f["dependents"])  # ascending

        def key_max_unblock(t):
            f = feats[t]
            return (-f["dependents"], f["holds_count"], f["wait_age"])  # dependents desc

        def key_most_resources(t):
            f = feats[t]
            return (-f["holds_count"], -f["dependents"], f["wait_age"])  # holds desc

        if policy == "max_unblock":
            victim = sorted(candidates, key=key_max_unblock)[0]
            rationale = "Chosen to unblock the most waiting threads."
        elif policy == "most_resources":
            victim = sorted(candidates, key=key_most_resources)[0]
            rationale = "Chosen because it holds the most resources"
        else:
            victim = sorted(candidates, key=key_min_cost)[0]
            rationale = "Chosen for minimal cost."

        exp = {
            "policy": policy,
            "rationale": rationale,
            "candidates": {
                t: {
                    "holds_count": feats[t]["holds_count"],
                    "dependents": feats[t]["dependents"],
                    "wait_age_sec": round(feats[t]["wait_age"], 3),
                } for t in candidates
            },
            "winner": victim,
            "winner_features": {
                **feats[victim],
                "wait_age_sec": round(feats[victim]["wait_age"], 3),
            },
        }
        return victim, exp

    # Backwards-compatible method used elsewhere
    @staticmethod
    def choose_victim(rm: ResourceManager, candidates: List[str]) -> str:
        v, _ = DeadlockDetector.choose_victim_explained(rm, candidates, policy="most_resources")
        return v

class Worker(threading.Thread):
    def __init__(self, tid: str, rm: ResourceManager, logger: Logger):
        super().__init__(daemon=True)
        self.tid = tid
        self.rm = rm
        self.logger = logger
        self.running = True
    def run(self):
        while self.running:
            all_res = list(self.rm.resources.keys())
            k = random.randint(1, min(3, len(all_res)))
            req = random.sample(all_res, k)
            self.logger.log("INFO", "REQUEST", self.tid, details=str(req))
            ok = self.rm.acquire(req, self.tid, shuffle=True)
            if not ok:
                self.logger.log("WARN", "ABORTED", self.tid, details="aborted while waiting")
                break
            t0 = time.time(); hold = random.uniform(0.5, 1.5)
            while time.time() - t0 < hold:
                if self.tid in self.rm.aborted: break
                time.sleep(0.05)
            self.rm.release_all(self.tid)
            time.sleep(random.uniform(0.1, 0.4))

class Simulation:
    def __init__(self, logger: Logger, threads=10, resources=6):
        self.logger = logger
        self.rm = ResourceManager([f"R{i}" for i in range(resources)], self.logger)
        self.workers: List[Worker] = [Worker(f"T{i}", self.rm, self.logger) for i in range(threads)]
        self.pending_deadlock: Optional[dict] = None
        self.suppress_deadlock_until: float = 0.0
        def on_deadlock(info):
            # Ignore detector events during the suppression window
            if time.time() < self.suppress_deadlock_until:
                return
            self.pending_deadlock = info
        self.detector = DeadlockDetector(self.rm, self.logger, on_deadlock=on_deadlock)
        self.running = False
    def start(self):
        if self.running: return
        self.running = True
        for w in self.workers: w.start()
        self.detector.start()
        self.logger.log("INFO", "SIM_START", details=f"threads={len(self.workers)} resources={len(self.rm.resources)}")
    def stop(self):
        if not self.running: return
        self.detector.running = False
        for w in self.workers: w.running = False
        with self.rm.cv:
            self.rm.cv.notify_all()
        self.running = False
        self.logger.log("INFO", "SIM_STOP")
    def metrics(self):
        p = psutil.Process()
        return {
            "cpu": p.cpu_percent(interval=0.2),
            "mem_mb": p.memory_info().rss / (1024*1024),
            "threads_alive": sum(1 for w in self.workers if w.is_alive()),
            "resources": len(self.rm.resources),
        }

# ----------------------------- Streamlit UI -----------------------------
st.set_page_config(page_title="Real-Time Deadlock Simulator", layout="wide")

if "last_resolved" not in st.session_state:
    st.session_state.last_resolved = None

# Persist a single Logger across resets so logs aren't lost
def get_logger() -> Logger:
    if "shared_logger" not in st.session_state:
        # Default: unbounded buffer
        st.session_state.shared_logger = Logger(max_buffer_size=None)
    return st.session_state.shared_logger

def get_sim() -> Simulation:
    if "sim" not in st.session_state:
        st.session_state.sim = Simulation(logger=get_logger(), threads=10, resources=6)
    return st.session_state.sim

def resource_snapshot(sim: Simulation):
    rm = sim.rm
    with rm.cv:
        rows = []
        free = in_use = waiting = 0
        for r_id, stt in rm.resources.items():
            holder = stt.holder or "-"
            q = list(stt.wait_queue)
            qlen = len(q)
            if stt.holder is None:
                free += 1; left = 1
            else:
                in_use += 1; left = 0
            waiting += qlen
            rows.append({
                "Resource": r_id,
                "Holder": holder,
                "Queue": ",".join(q) if q else "",
                "Left": left,
            })
    totals = {"free": free, "in_use": in_use, "waiting": waiting, "total": len(rm.resources)}
    return rows, totals

def victim_wait_info(sim: Simulation, victim_tid: str):
    rm = sim.rm
    with rm.cv:
        r = rm.waiting_for.get(victim_tid)
        if not r:
            return None
        holder = rm.resources[r].holder
        return {"victim": victim_tid, "resource": r, "holder": holder}

def draw_wfg(sim, ax):
    g = sim.rm.build_wfg()
    spacing = 8.0
    curvature = 0.18
    n = max(len(g), 1)
    k = spacing / np.sqrt(n)
    pos = nx.spring_layout(g, k=k, iterations=100, seed=3)

    nx.draw_networkx_nodes(g, pos, ax=ax, node_size=1000)
    nx.draw_networkx_labels(
        g, pos, ax=ax, font_size=11,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7)
    )
    nx.draw_networkx_edges(
        g, pos, ax=ax, arrows=True, width=1.2, arrowsize=18,
        connectionstyle=f"arc3,rad={curvature}"
    )

    ax.set_title("Wait-For Graph")
    ax.axis("off")
    ax.margins(0.20)

    info = sim.pending_deadlock
    if not info:
        return
    try:
        cyc_nodes = set(info.get("nodes", [])) & set(g.nodes)
        cyc_edges = [(u, v) for (u, v) in (tuple(e) for e in info.get("edges", [])) if g.has_edge(u, v)]
        if cyc_nodes:
            nx.draw_networkx_nodes(g, pos, nodelist=list(cyc_nodes), ax=ax, node_size=1200, node_color="red", alpha=0.30)
        if cyc_edges:
            nx.draw_networkx_edges(g, pos, edgelist=cyc_edges, ax=ax, width=1.2, arrows=True, arrowsize=12, connectionstyle=f"arc3,rad={curvature}")
    except Exception:
        pass

def last_events(sim: Simulation, limit: Optional[int] = None, only_cycle: Optional[Set[str]] = None):
    with sim.logger.lock:
        items = list(sim.logger.buffer)

    rows = []
    for rec in items:
        if only_cycle and rec.tid and rec.tid not in only_cycle:
            continue
        rows.append(rec.as_dict())

    if limit:
        rows = rows[-limit:]
    return rows

# ----------------------------- UI: Controls -----------------------------
st.sidebar.header("Controls")
sim = get_sim()

colA, colB = st.sidebar.columns(2)
with colA:
    if st.button("Start", use_container_width=True):
        if not sim.running:
            sim.start()
with colB:
    if st.button("Stop", use_container_width=True):
        sim.stop()

# Allow configuring thread/resource counts on reset
threads = st.sidebar.slider("Threads", 2, 40, len(sim.workers), 1)
resources = st.sidebar.slider("Resources", 2, 12, len(sim.rm.resources), 1)

# Optional: let the user cap the buffer if desired (default None = unbounded)
buf_cap = st.sidebar.selectbox("Log buffer cap", ["Unbounded", 5_000, 20_000, 100_000], index=0)
if buf_cap != "Unbounded":
    get_logger().max_buffer_size = int(buf_cap)
else:
    get_logger().max_buffer_size = None

if st.sidebar.button("Reset", use_container_width=True):
    # Preserve the shared logger so logs persist across resets
    if sim.running:
        sim.stop()
    st.session_state.sim = Simulation(logger=get_logger(), threads=threads, resources=resources)
    sim = st.session_state.sim

# --- Resources panel -------------------------------------------------------
st.sidebar.markdown("### Resources")
r_rows, r_tot = resource_snapshot(sim)
c1, c2 = st.sidebar.columns(2)
c1.metric("Free", r_tot["free"])
c2.metric("In use", r_tot["in_use"])
st.sidebar.metric("Waiting", r_tot["waiting"])
st.sidebar.dataframe(r_rows, use_container_width=True, height=220)

st.sidebar.markdown("---")

# Deadlock status messages (resolved vs detected) + victim policy
if sim.pending_deadlock:
    st.sidebar.error("Deadlock detected!")
    edetail = sim.pending_deadlock.get("edge_details", [])
    deadlock_type = sim.pending_deadlock.get("deadlock_type", "cycle")
    self_waiters = sim.pending_deadlock.get("self_waiters", [])
    victims = sim.pending_deadlock.get("nodes", []) or []

    if deadlock_type == "self-wait":
        st.sidebar.markdown("**Type:** Self-wait (re-entrant request)")
        for w in self_waiters:
            r = next((e.get("resource") for e in edetail if e["waiter"] == w and e["holder"] == w), None)
            st.sidebar.markdown(f"- **{w}** requested **{r or '(unknown)'}** which it already holds.")
        st.sidebar.info("Re-requests are treated as granted to prevent blocking.")
    else:
        st.sidebar.markdown("**Cycle edges**")
        for e in edetail:
            st.sidebar.markdown(f"- **{e['waiter']}** → **{e.get('resource') or '(unknown)'}** → **{e['holder']}**")

    # Victim policy selector + auto-pick with explanation
    policy = st.sidebar.selectbox("Victim policy", ["min_cost", "max_unblock", "most_resources"], index=0)
    best_victim, victim_exp = (DeadlockDetector.choose_victim_explained(sim.rm, victims, policy=policy)
                               if victims else (None, None))

    # Allow manual override but default to auto-picked
    if victims:
        default_index = victims.index(best_victim) if best_victim in victims else 0
        victim = st.sidebar.selectbox("Victim", victims, index=default_index)
    else:
        victim = None

    if victim_exp:
        st.sidebar.markdown("**Why this victim?**")
        st.sidebar.markdown(f"- **Policy:** `{victim_exp['policy']}` — {victim_exp['rationale']}")
        wf = victim_exp["winner_features"]
        st.sidebar.markdown(
            f"- **Holds:** {wf['holds_count']}  \n"
            f"- **Would unblock:** {wf['dependents']} thread(s)  \n"
            f"- **Wait age:** {wf['wait_age_sec']}s"
        )
        with st.sidebar.expander("Compare all candidates"):
            for t, f in victim_exp["candidates"].items():
                mark = "✅ " if t == victim_exp["winner"] else "• "
                st.markdown(f"{mark}**{t}** — holds={f['holds_count']}, dependents={f['dependents']}, wait_age={f['wait_age_sec']}s")

    vinfo = victim_wait_info(sim, victim) if victim else None
    if vinfo:
        st.sidebar.markdown(
            f"Victim **{vinfo['victim']}** is waiting for **{vinfo['resource']}** "
        )

    if st.sidebar.button("Resolve", use_container_width=True, type="primary"):
        chosen = victim if victim else best_victim
        # Log the reasoning too
        sim.logger.log("ERROR", "RESOLUTION",
                       tid=chosen,
                       details=f"terminate via UI; policy={policy}; explanation={victim_exp}")
        sim.rm.abort(chosen)
        sim.pending_deadlock = None
        st.session_state.last_resolved = time.time()
        # Suppress fresh detector events briefly so UI shows 'resolved'
        sim.suppress_deadlock_until = time.time() + 2.0

elif st.session_state.last_resolved:
    # Show "Deadlock resolved" briefly, then clear
    st.sidebar.success("Deadlock resolved")
    if time.time() - st.session_state.last_resolved > 5:
        st.session_state.last_resolved = None

# ----------------------------- Main area -----------------------------
m = sim.metrics()
m1, m2, m3, m4 = st.columns(4)
m1.metric("CPU", f"{m['cpu']:.0f}%")
m2.metric("Memory (MB)", f"{m['mem_mb']:.1f}")
m3.metric("Threads alive", f"{m['threads_alive']}")
m4.metric("Resources", f"{m['resources']}")

left, right = st.columns([2,1])

with left:
    fig, ax = plt.subplots(figsize=(9, 6.5))
    draw_wfg(sim, ax)
    st.pyplot(fig, clear_figure=True)

with right:
    st.subheader("System Log")

    only_cycle = set(sim.pending_deadlock["nodes"]) if sim.pending_deadlock else None
    # Default to showing EVERYTHING; user can opt into filtering
    show_cycle_only = st.checkbox("Show only processes in detected cycle", value=False)

    row_choice = st.selectbox("Rows to show", ["All", 50, 200, 1000], index=0)
    limit = None if row_choice == "All" else int(row_choice)

    rows = last_events(sim, limit=limit, only_cycle=only_cycle if show_cycle_only else None)
    st.dataframe(rows, use_container_width=True, height=420)

    # Download full buffer to avoid any loss
    try:
        import pandas as pd
        with sim.logger.lock:
            full_df = pd.DataFrame([r.as_dict() for r in sim.logger.buffer])
        st.download_button(
            "Download all logs (CSV)",
            data=full_df.to_csv(index=False).encode(),
            file_name="logs.csv",
            mime="text/csv",
            use_container_width=True
        )
    except Exception:
        pass

    if sim.pending_deadlock:
        st.subheader("Deadlock Explanation")
        edetail = sim.pending_deadlock.get("edge_details", [])
        deadlock_type = sim.pending_deadlock.get("deadlock_type", "cycle")
        self_waiters = sim.pending_deadlock.get("self_waiters", [])
        nodes = sim.pending_deadlock.get("nodes", [])

        if deadlock_type == "self-wait":
            st.markdown("**Type:** Self-wait (re-entrant request)**")
            for w in self_waiters:
                r = next((e.get("resource") for e in edetail if e["waiter"] == w and e["holder"] == w), None)
                st.markdown(f"- **{w}** requested **{r or '(unknown)'}** which it already holds.")
        else:
            st.markdown("**Cycle edges**")
            for e in edetail:
                st.markdown(f"- **{e['waiter']}** is waiting for **{e.get('resource') or '(unknown)'}** held by **{e['holder']}**")

if sim.pending_deadlock:
    pass

if sim.running:
    time.sleep(1.0)
    st.rerun()


# Show the Overview of the number of deadlocks happened
# Show the Thread CPU Usage
# Show the Memory Usage
# Resources in detailed
# Timestamp
