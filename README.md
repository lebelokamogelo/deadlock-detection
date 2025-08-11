# Real-Time Deadlock Simulator

A Streamlit-based simulation of resource allocation, deadlock detection, and resolution in a multi-threaded system.

## Features

* **Multi-threaded simulation** of processes competing for shared resources.
* **Resource Manager** tracks allocations, wait queues, and builds:

  * **Wait-For Graph (WFG)** for cycle detection.
* **Deadlock detection** using NetworkX cycle detection.
* **Deadlock resolution** by **process termination**, with multiple victim-selection policies:

  * `min_cost` – aborts process with minimal rollback cost.
  * `max_unblock` – aborts process that unblocks the most others.
  * `most_resources` – aborts process holding the most resources.
* **Interactive Streamlit UI**:

  * Live WFG visualization with detected cycles highlighted.
  * Resource and system metrics.
  * Detailed event logs and deadlock explanations.
  * Control simulation threads/resources and reset.

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the App

```bash
streamlit run main.py
```

## Usage

1. Use the **Start** button to run the simulation.
2. Adjust **Threads** and **Resources** sliders in the sidebar.
3. When a deadlock is detected:

   * View cycle edges and victim-selection explanation.
   * Choose a victim manually or use the suggested one.
   * Click **Resolve** to terminate and release resources.
4. View logs and metrics in real time.

## Notes

* The simulation uses random resource requests to trigger deadlocks.
* Re-entrant resource requests are granted automatically to avoid false self-waits.
* Victim selection is explained in the sidebar for transparency.
