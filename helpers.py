import time, threading
from typing import List, Tuple
import networkx as nx
from core import ResourceManager, Logger

class DeadlockDetector(threading.Thread):
    def __init__(self, rm: ResourceManager, logger: Logger, interval: float = 0.5, on_deadlock=None):
        super().__init__(daemon=True)
        self.rm = rm
        self.logger = logger
        self.interval = interval
        self.running = True
        self.on_deadlock = on_deadlock
        self.total_deadlocks = 0
        self.currently_in_deadlock = False  # prevent double counting the same cycle

    def run(self):
        while self.running:
            g = self.rm.build_wfg()
            try:
                cyc = nx.find_cycle(g, orientation="original")
                # Only fire once per stuck period
                if not self.currently_in_deadlock:
                    edges = [(u, v) for u, v, _ in cyc]
                    nodes = list({u for u, v in edges} | {v for u, v in edges})
                    self_waiters = [u for (u, v) in edges if u == v]
                    deadlock_type = "self-wait" if self_waiters else "cycle"
                    edge_details = [{"waiter": u, "holder": v, "resource": self.rm.edge_resource(u, v)} for (u, v) in edges]
                    self.logger.log(
                        "ERROR",
                        "DEADLOCK_DETECTED",
                        tid=",".join(nodes),  # all threads in the cycle
                        res=",".join(r for r in {e["resource"] for e in edge_details if e["resource"]}),
                        details=f"type={deadlock_type} cycle={'→'.join(nodes)}"
                    )
                    self.total_deadlocks += 1
                    self.currently_in_deadlock = True
                    if self.on_deadlock:
                        self.on_deadlock({
                            "nodes": nodes,
                            "edges": edges,
                            "edge_details": edge_details,
                            "deadlock_type": deadlock_type,
                            "self_waiters": self_waiters
                        })
            except nx.exception.NetworkXNoCycle:
                # no change; we only clear the "currently_in_deadlock" when UI resolves
                pass
            time.sleep(self.interval)

    # -------- Victim selection with explanation --------
    @staticmethod
    def _features_for(rm: ResourceManager, tid: str) -> dict:
        holds = list(rm.held_by_thread.get(tid, set()))
        holds_count = len(holds)
        dependents = 0
        with rm.cv:
            for r in holds:
                stt = rm.resources[r]
                dependents += len(stt.wait_queue)
        ws = rm.waiting_since.get(tid)
        wait_age = 0.0 if ws is None else max(0.0, time.time() - ws)
        return {"holds_count": holds_count, "dependents": dependents, "wait_age": wait_age}

    @staticmethod
    def choose_victim_explained(rm: ResourceManager, candidates: List[str], policy: str = "min_cost") -> Tuple[str, dict]:
        feats = {t: DeadlockDetector._features_for(rm, t) for t in candidates}

        def key_min_cost(t):
            f = feats[t]
            return (f["holds_count"], f["wait_age"], f["dependents"])

        def key_max_unblock(t):
            f = feats[t]
            return (-f["dependents"], f["holds_count"], f["wait_age"])

        def key_most_resources(t):
            f = feats[t]
            return (-f["holds_count"], -f["dependents"], f["wait_age"])

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

    @staticmethod
    def choose_victim(rm: ResourceManager, candidates: List[str]) -> str:
        v, _ = DeadlockDetector.choose_victim_explained(rm, candidates, policy="most_resources")
        return v
