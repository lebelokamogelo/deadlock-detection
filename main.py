import time
from typing import Optional, Set
import numpy as np
import networkx as nx
import streamlit as st
import matplotlib.pyplot as plt
from core import Logger
from simulation import Simulation, RESOURCE_LAYOUT
from helpers import DeadlockDetector

# ----------------------------- Helpers for UI -----------------------------
st.set_page_config(page_title="Real-Time Deadlock Simulator", layout="wide")

if "last_resolved" not in st.session_state:
    st.session_state.last_resolved = None

def get_logger() -> Logger:
    if "shared_logger" not in st.session_state:
        st.session_state.shared_logger = Logger(max_buffer_size=None)
    return st.session_state.shared_logger

def get_sim() -> Simulation:
    if "sim" not in st.session_state:
        st.session_state.sim = Simulation(logger=get_logger(), threads=10, resources=None)
    return st.session_state.sim

def resource_snapshot(sim: Simulation):
    """
    Capture comprehensive snapshot of current resource allocation state.

    Analyzes all resources to determine usage patterns, wait queues,
    and generates summary statistics by resource type for dashboard display.

    """

    rm = sim.rm
    with rm.cv:
        by_type = {k: {"total": v, "held": 0, "waiting": 0, "units": []} for k, v in RESOURCE_LAYOUT.items()}
        rows = []
        for r_id, stt in rm.resources.items():
            rtype = r_id.split("-")[0]
            holder = stt.holder or "-"
            q = list(stt.wait_queue)
            if stt.holder is not None:
                by_type[rtype]["held"] += 1
            by_type[rtype]["waiting"] += len(q)
            by_type[rtype]["units"].append({"unit": r_id, "holder": holder, "queue": ",".join(q) if q else ""})
            rows.append({"time": "", "level": "", "event": "", "Resource": r_id, "Holder": holder, "Queue": ",".join(q) if q else ""})
        totals = {
            name: {
                "total": info["total"],
                "held": info["held"],
                "free": info["total"] - info["held"],
                "waiting": info["waiting"],
            } for name, info in by_type.items()
        }
    return rows, totals, by_type

def victim_wait_info(sim: Simulation, victim_tid: str):
    """
    Extract wait relationship details for a potential deadlock victim.

    Determines what resource the victim thread is waiting for and which
    thread currently holds that resource. Used for deadlock explanation
    and resolution planning.

    """
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

# ----------------------------- Sidebar Controls ---------------------------
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

threads = st.sidebar.slider("Threads", 2, 40, len(sim.workers), 1)

buf_cap = st.sidebar.selectbox("Log buffer cap", ["Unbounded", 5_000, 20_000, 100_000], index=0)
if buf_cap != "Unbounded":
    get_logger().max_buffer_size = int(buf_cap)
else:
    get_logger().max_buffer_size = None

if st.sidebar.button("Reset", use_container_width=True):
    if sim.running:
        sim.stop()
    cap = None if buf_cap == "Unbounded" else int(buf_cap)
    st.session_state.shared_logger = Logger(max_buffer_size=cap)
    st.session_state.sim = Simulation(
        logger=st.session_state.shared_logger,
        threads=threads,
        resources=None
    )
    sim = st.session_state.sim
    st.session_state.last_resolved = None
    sim.logger.log("INFO", "RESET", details=f"threads={threads} resources={len(sim.rm.resources)}", force=True)

# ----------------------------- Overview + Simulation Tabs -----------------
overview_tab, sim_tab = st.tabs(["Overview", "Simulation"])

# ----------------------------- OVERVIEW -----------------------------------
with overview_tab:
    m = sim.metrics()
    _, totals, _ = resource_snapshot(sim)

    # Metrics (no timestamp card here; timestamps live in the log)
    top1, top2, top3, top4 = st.columns(4)
    with top1:
        st.metric("Deadlocks Detected", f"{m['deadlocks_total']}")
    with top2:
        st.metric("Deadlocks Resolved", f"{m['deadlocks_resolved']}")
    with top3:
        st.metric("CPU", f"{m['cpu']:.0f}%")
    with top4:
        st.metric("Memory (MB)", f"{m['mem_mb']:.1f}")

    # Current deadlock status
    if sim.pending_deadlock:
        st.error("Deadlock detected")
    else:
        st.info("Deadlock detected: None")

    st.markdown("### Resources")
    rc1, rc2, rc3, rc4 = st.columns(4)
    cols = [rc1, rc2, rc3, rc4]
    idx = 0
    for name in ["DB", "FILE", "NET", "LOCK"]:
        c = cols[idx]; idx += 1
        info = totals[name]
        used = info["held"]; total = info["total"]; free = info["free"]; waiting = info["waiting"]
        usage = (used / total) if total else 0.0
        with c:
            st.write(f"**{name}**")
            st.progress(min(max(usage, 0.0), 1.0), text=f"Usage: {used}/{total}")
            st.caption(f"Free: {free} | Waiting: {waiting}")

    # Thread waiting / resources table
    st.markdown("### Threads")
    wait_rows = []
    with sim.rm.cv:
        for t, r in sim.rm.waiting_for.items():
            if r:
                holder = sim.rm.resources[r].holder
                wait_rows.append({
                    "Thread": t,
                    "Waiting For": r,
                    "Held By": holder or "-",
                })
    if wait_rows:
        st.dataframe(wait_rows, use_container_width=True, height=240)
    else:
        st.caption("No threads are currently waiting.")

# ----------------------------- SIMULATION ---------------------------------
with sim_tab:
    m = sim.metrics()
    m1, m2, m4 = st.columns(3)
    m1.metric("CPU", f"{m['cpu']:.0f}%")
    m2.metric("Memory (MB)", f"{m['mem_mb']:.1f}")
    m4.metric("Resources", f"{m['resources']}")

    left, right = st.columns([2,1])

    with left:
        fig, ax = plt.subplots(figsize=(9, 6.5))
        draw_wfg(sim, ax)
        st.pyplot(fig, clear_figure=True)

    with right:
        st.subheader("System Log")

        only_cycle = set(sim.pending_deadlock["nodes"]) if sim.pending_deadlock else None
        show_cycle_only = st.checkbox("Show only processes in detected cycle", value=False)

        row_choice = st.selectbox("Rows to show", ["All", 50, 200, 1000], index=0)
        limit = None if row_choice == "All" else int(row_choice)

        rows = last_events(sim, limit=limit, only_cycle=only_cycle if show_cycle_only else None)
        st.dataframe(rows, use_container_width=True, height=420)

        try:
            import pandas as pd
            with sim.logger.lock:
                full_df = pd.DataFrame([r.as_dict() for r in sim.logger.buffer])
            st.download_button(
                "Download as CSV",
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
            if deadlock_type == "self-wait":
                st.markdown("**Type:** Self-wait (re-entrant request)**")
                for w in self_waiters:
                    r = next((e.get("resource") for e in edetail if e["waiter"] == w and e["holder"] == w), None)
                    st.markdown(f"- **{w}** requested **{r or '(unknown)'}** which it already holds.")
            else:
                st.markdown("**Cycle edges**")
                for e in edetail:
                    st.markdown(f"- **{e['waiter']}** is waiting for **{e.get('resource') or '(unknown)'}** held by **{e['holder']}**")

# ----------------------------- Deadlock sidebar actions --------------------
st.sidebar.markdown("---")
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

    policy = st.sidebar.selectbox("Victim policy", ["min_cost", "max_unblock", "most_resources"], index=0)
    best_victim, victim_exp = (DeadlockDetector.choose_victim_explained(sim.rm, victims, policy=policy)
                               if victims else (None, None))

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
        # Log termination & resolution even while paused
        sim.logger.log("ERROR", "RESOLUTION",
                       tid=chosen,
                       details=f"terminate via UI; policy={policy}; explanation={victim_exp}",
                       force=True)
        sim.rm.abort(chosen)

        # Mark resolved counts & clear deadlock status
        sim.pending_deadlock = None
        sim.resolved_count += 1
        sim.detector.currently_in_deadlock = False

        st.session_state.last_resolved = time.time()
        # Resume logging now that the deadlock is resolved
        sim.logger.resume()
        # Suppress fresh detector events briefly so UI shows 'resolved'
        sim.suppress_deadlock_until = time.time() + 2.0

elif st.session_state.last_resolved:
    st.sidebar.success("Deadlock resolved")
    if time.time() - st.session_state.last_resolved > 5:
        st.session_state.last_resolved = None

# Main-loop tick
if sim.running:
    time.sleep(1.0)
    st.rerun()
