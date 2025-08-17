import time, random, threading
from typing import Dict, List, Optional
import psutil
from core import Logger, ResourceManager
from helpers import DeadlockDetector

class Worker(threading.Thread):
    def __init__(self, tid: str, rm: ResourceManager, logger: Logger):
        super().__init__(daemon=True)
        self.tid = tid
        self.rm = rm
        self.logger = logger
        self.running = True
    def run(self):
        """
        Main worker loop - request resources, work, release, repeat.

        Each iteration:
        1. Randomly selects 1-3 resources to request
        2. Attempts to acquire all selected resources
        3. Holds resources for random duration (0.5-1.5 seconds)
        4. Releases all resources
        5. Sleeps briefly before next iteration
        """
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

# ----------------------------- Resource Layout ---------------------
RESOURCE_LAYOUT = {
    "DB": 3,
    "FILE": 2,
    "NET": 4,
    "LOCK": 2,
}
def expand_resource_ids(layout: Dict[str, int]) -> List[str]:
    ids: List[str] = []
    for name, cap in layout.items():
        for i in range(1, cap+1):
            ids.append(f"{name}-{i}")
    return ids

# ----------------------------- Simulation -------------------------
class Simulation:
    """
    Main simulation orchestrator managing all system components.

    Coordinates worker threads, resource management, deadlock detection,
    and provides interfaces for system control and monitoring. Handles
    the complete lifecycle from startup through deadlock resolution.
    """
    def __init__(self, logger: Logger, threads=10, resources=None):
        self.logger = logger
        res_ids = expand_resource_ids(RESOURCE_LAYOUT)
        self.rm = ResourceManager(res_ids, self.logger)
        self.workers: List[Worker] = [Worker(f"T{i}", self.rm, self.logger) for i in range(threads)]
        self.pending_deadlock: Optional[dict] = None
        self.suppress_deadlock_until: float = 0.0
        self.resolved_count: int = 0  # number of user-resolved deadlocks

        def on_deadlock(info):
            """
            Callback invoked when deadlock detector finds a cycle.

            Pauses logging to freeze system state for analysis and resolution.
            Respects suppression periods to avoid duplicate notifications.
            """

            # Ignore during suppression
            if time.time() < self.suppress_deadlock_until:
                return
            self.pending_deadlock = info
            # Pause logs immediately so overview stops counting after detection
            self.logger.pause()

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
            "deadlocks_total": getattr(self.detector, "total_deadlocks", 0),
            "deadlocks_resolved": self.resolved_count,
        }
