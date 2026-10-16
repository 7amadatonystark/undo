"""
try_agent.py — run ONE scenario with the real AI agent a few times, WITH UNDO, and watch it.

    python try_agent.py                        # clean_test_accounts, 3 runs
    python try_agent.py free_up_space 3

It prints every tool call and UNDO decision (verbose), then a short summary. It uses the same
run code as run_benchmark.py, so what you see here is exactly what the benchmark records.
Runs are saved to results/practice_runs.jsonl in a separate batch ("try"), so they never mix
into the benchmark's scoreboard.
"""

import sys

import config
from run_benchmark import run_one, save
from scenarios import SCENARIOS


def find_scenario(name):
    for s in SCENARIOS:
        if s["name"] == name:
            return s
    raise SystemExit(f"No scenario named {name!r}. Try one of: "
                     + ", ".join(s["name"] for s in SCENARIOS))


def main():
    sys.stdout.reconfigure(errors="replace")      # never crash on characters the terminal can't show
    name = sys.argv[1] if len(sys.argv) > 1 else "clean_test_accounts"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    scenario = find_scenario(name)
    print(f"Scenario: {name}   Model: {config.AGENT_MODEL}   Runs: {runs} (WITH UNDO)")
    print(f"Task: {scenario['task']}")

    records = []
    for n in range(1, runs + 1):
        print(f"\n=============== Run {n} of {runs} ===============")
        record = run_one(scenario, "undo", run_number=n, verbose=True, batch="try")
        save(record)
        records.append(record)

    print("\n=============== Summary ===============")
    print(f"{'RUN':>3}  {'STEPS':>5}  {'STOPPED':8s} {'DECISION':9s} {'OUTCOME':10s} TOOLS USED")
    for r in records:
        tools = ", ".join(s["tool"] for s in r["steps"]) or "(none)"
        print(f"{r['run']:>3}  {len(r['steps']):>5}  {r['stop_reason']:8s} {r['decision']:9s} "
              f"{r['outcome']:10s} {tools}")


if __name__ == "__main__":
    main()
