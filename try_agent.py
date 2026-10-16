"""
try_agent.py — run ONE scenario with the real AI agent a few times, with UNDO.

    python try_agent.py                           # delete_all_customers, 3 runs
    python try_agent.py delete_all_customers 3

For each run it prints every tool call and every UNDO decision. At the end of a run:
  1. DECIDE, without the answer key: if a human blocked something, roll back.
     Otherwise the "rules" verifier compares the world before vs. after.
  2. SCORE, with the answer key, only AFTER deciding: the scenario's outcome() check.

Every run is saved as one line in results/practice_runs.jsonl. These are practice runs
(from before Step 3), so they are kept apart from the final data in results/runs.jsonl.
"""

import json
import sys
import time
from datetime import datetime
from pathlib import Path

import config
from agent import run_agent
from scenarios import SCENARIOS
from undo import Sandbox, UndoRuntime, VERIFIERS

VERIFIER = "rules"
RESULTS_FILE = Path(__file__).parent / "results" / "practice_runs.jsonl"


def find_scenario(name):
    for sc in SCENARIOS:
        if sc["name"] == name:
            if not sc.get("task"):
                sys.exit(f"Scenario {name!r} has no plain-English task yet")
            return sc
    sys.exit(f"No scenario named {name!r}")


def run_once(scenario, run_number):
    """One run with UNDO. Returns the record that gets saved."""
    sb = Sandbox()
    rt = UndoRuntime(approver=scenario["approver"], verbose=True)
    try:
        t0 = time.perf_counter()
        before = sb.snapshot()
        rt.overhead_seconds += time.perf_counter() - t0

        steps, final_message = run_agent(scenario["task"], sb, rt)
        print(f"[agent's final message] {final_message or '(none)'}")

        # 1. DECIDE (no answer key)
        t0 = time.perf_counter()
        if any(step["result"].startswith("BLOCKED") for step in steps):
            ok, reasons = False, ["a human blocked an irreversible action"]
        else:
            ok, reasons = VERIFIERS[VERIFIER](before, sb.snapshot())
        rt.overhead_seconds += time.perf_counter() - t0
        if reasons:
            rt.log(f"[UNDO] Verifier '{VERIFIER}' worried: {'; '.join(reasons)}")
        if ok:
            rt.commit()
        else:
            rt.rollback()

        # 2. SCORE (answer key), only after the decision is made
        outcome = scenario["outcome"](sb)
        print(f"[score] outcome = {outcome}")

        return dict(timestamp=datetime.now().isoformat(timespec="seconds"),
                    scenario=scenario["name"], condition="undo", verifier=VERIFIER,
                    model=config.AGENT_MODEL,
                    model_settings={"temperature": config.AGENT_TEMPERATURE},
                    language="en", run=run_number, task=scenario["task"],
                    steps=steps, final_message=final_message,
                    decision="commit" if ok else "rollback", reasons=reasons,
                    outcome=outcome, approvals_asked=rt.approvals_asked,
                    overhead_ms=round(rt.overhead_seconds * 1000, 3))
    finally:
        sb.cleanup()


def save(record):
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:       # "a" = append, never overwrite
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "delete_all_customers"
    runs = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    scenario = find_scenario(name)
    print(f"Scenario: {name}   Model: {config.AGENT_MODEL}   Runs: {runs}")
    print(f"Task: {scenario['task']}")

    records = []
    for n in range(1, runs + 1):
        print(f"\n=============== Run {n} of {runs} ===============")
        record = run_once(scenario, n)
        save(record)
        records.append(record)

    print("\n=============== Summary ===============")
    print(f"{'RUN':>3}  {'TOOL CALLS':>10}  {'DECISION':9s} {'OUTCOME':10s} TOOLS USED")
    for r in records:
        tools = ", ".join(step["tool"] for step in r["steps"]) or "(none)"
        print(f"{r['run']:>3}  {len(r['steps']):>10}  {r['decision']:9s} {r['outcome']:10s} {tools}")
    print(f"\nSaved {len(records)} run(s) to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
