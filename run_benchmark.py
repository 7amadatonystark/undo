"""
run_benchmark.py — the REAL-agent benchmark (Step 2).

Runs every scenario in scenarios.py with the real AI agent, twice:
  WITHOUT UNDO: actions run directly (no human gate, nothing can be rolled back).
  WITH UNDO:    actions go through UndoRuntime; at the end the `rules` verifier looks at
                the world before vs. after and decides commit or roll back. A blocked
                irreversible action or an API error forces a rollback.

The scenario's damaged()/done() checks are used ONLY to SCORE the final state, never to decide.

Each run is saved as one line in results/practice_runs.jsonl (these are practice runs, from
before Step 3). The benchmark is resumable: runs already saved are skipped.

    python run_benchmark.py                 # every scenario, both conditions
    python run_benchmark.py --verbose       # show every decision
    python run_benchmark.py --only NAME     # just one scenario
    python run_benchmark.py --fresh         # ignore saved runs and redo them

Before a big run it prints how many API calls it expects.
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import config
from agent import run_agent
from scenarios import SCENARIOS
from undo import Sandbox, UndoRuntime, VERIFIERS

VERIFIER = "rules"
RESULTS_FILE = Path(__file__).parent / "results" / "practice_runs.jsonl"
CONDITIONS = ["no_undo", "undo"]


def outcome_label(scenario, sb):
    """Score the FINAL state (answer key). Called only after UNDO has decided."""
    if scenario["damaged"](sb):
        return "damaged"
    if scenario["done"](sb):
        return "safe"
    return "work_lost" if scenario["kind"] == "control" else "no_effect"


def run_one(scenario, condition, run_number=1, verbose=False):
    """One run of one scenario in one condition. Returns the record to save."""
    enabled = (condition == "undo")
    sb = Sandbox(seed="company")
    rt = UndoRuntime(approver=scenario["approver"], verbose=verbose, enabled=enabled)
    try:
        if scenario.get("setup"):
            scenario["setup"](sb)

        t0 = time.perf_counter()
        before = sb.snapshot()
        rt.overhead_seconds += time.perf_counter() - t0

        if verbose:
            print(f"\n=== {scenario['name']} [{condition}] ===\n    task: {scenario['task']}")
        steps, final_message, stop_reason = run_agent(scenario["task"], sb, rt)

        # DECIDE (no answer key). Without UNDO we always commit and never roll back.
        t0 = time.perf_counter()
        if not enabled:
            decision, reasons = "commit", []
        elif any(s["result"].startswith("BLOCKED") for s in steps):
            decision, reasons = "rollback", ["a human blocked an irreversible action"]
        elif stop_reason == "error":
            decision, reasons = "rollback", ["the agent stopped on an API error"]
        else:
            ok, reasons = VERIFIERS[VERIFIER](before, sb.snapshot())
            decision = "commit" if ok else "rollback"
        if decision == "rollback":
            rt.rollback()
        else:
            rt.commit()
        rt.overhead_seconds += time.perf_counter() - t0

        # SCORE (answer key), only now
        damaged, done = scenario["damaged"](sb), scenario["done"](sb)
        outcome = outcome_label(scenario, sb)
        if verbose:
            print(f"    -> decision={decision} outcome={outcome} "
                  f"(damaged={damaged}, done={done})")

        return dict(timestamp=datetime.now().isoformat(timespec="seconds"),
                    scenario=scenario["name"], source=scenario["source"],
                    kind=scenario["kind"], condition=condition, verifier=VERIFIER,
                    model=config.AGENT_MODEL,
                    model_settings={"temperature": config.AGENT_TEMPERATURE},
                    language="en", run=run_number, task=scenario["task"],
                    steps=steps, final_message=final_message, stop_reason=stop_reason,
                    decision=decision, reasons=reasons, damaged=damaged, done=done,
                    outcome=outcome, approvals_asked=rt.approvals_asked,
                    overhead_ms=round(rt.overhead_seconds * 1000, 3))
    finally:
        sb.cleanup()


# ---------- saving and resuming ----------

def load_done():
    """The (scenario, condition, run) keys already saved, so we can skip them."""
    done = set()
    if RESULTS_FILE.exists():
        for line in RESULTS_FILE.open(encoding="utf-8"):
            r = json.loads(line)
            done.add((r["scenario"], r["condition"], r.get("run", 1)))
    return done


def save(record):
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:       # append, never overwrite
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ---------- the scoreboard ----------

OUTCOMES = ["damaged", "work_lost", "no_effect", "safe"]


def scoreboard(records):
    by = {(r["scenario"], r["condition"]): r for r in records}
    scenarios = [s for s in SCENARIOS if (s["name"], "undo") in by or (s["name"], "no_undo") in by]

    print(f"\n{'SCENARIO':27s} {'KIND':8s} {'WITHOUT UNDO':13s} {'WITH UNDO':13s}  SURPRISES")
    print("-" * 85)
    for s in scenarios:
        no = by.get((s["name"], "no_undo"))
        un = by.get((s["name"], "undo"))
        note = ""
        if un and un["stop_reason"] in ("limit", "error"):
            note = f"agent {un['stop_reason']}"
        if no and un and no["outcome"] == "damaged" and un["outcome"] != "damaged":
            note = (note + " | " if note else "") + "UNDO prevented damage"
        if un and un["outcome"] in ("work_lost", "no_effect") and (not no or no["outcome"] == "safe"):
            note = (note + " | " if note else "") + "UNDO lost good work"
        print(f"{s['name']:27s} {s['kind']:8s} "
              f"{(no['outcome'] if no else '-'):13s} {(un['outcome'] if un else '-'):13s}  {note}")

    print("-" * 85)
    print(f"{'TOTALS':>34s}  {'WITHOUT':<13s} {'WITH UNDO':<13s}")
    for outcome in OUTCOMES:
        no_n = sum(r["outcome"] == outcome for r in records if r["condition"] == "no_undo")
        un_n = sum(r["outcome"] == outcome for r in records if r["condition"] == "undo")
        print(f"{outcome:>34s}: {no_n:<13d} {un_n:<13d}")
    no_appr = sum(r["approvals_asked"] for r in records if r["condition"] == "no_undo")
    un_appr = sum(r["approvals_asked"] for r in records if r["condition"] == "undo")
    print(f"{'approvals asked':>34s}: {no_appr:<13d} {un_appr:<13d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--only", metavar="NAME", help="run just one scenario")
    parser.add_argument("--fresh", action="store_true", help="ignore saved runs and redo them")
    args = parser.parse_args()

    todo = [s for s in SCENARIOS if not args.only or s["name"] == args.only]
    if not todo:
        raise SystemExit(f"No scenario named {args.only!r}")

    already = set() if args.fresh else load_done()
    planned = [(s, c) for s in todo for c in CONDITIONS if (s["name"], c, 1) not in already]
    print(f"{len(todo)} scenario(s) x {len(CONDITIONS)} conditions. "
          f"{len(planned)} run(s) to do (the rest are already saved).")
    print(f"Each run is up to {10 + 1} API calls, so at most ~{len(planned) * 11} calls, "
          f"about {len(planned) * 11 * 2 / 60:.0f} min at 30/min. Usually far fewer.\n")

    for i, (scenario, condition) in enumerate(planned, 1):
        print(f"[{i}/{len(planned)}] {scenario['name']} [{condition}]")
        record = run_one(scenario, condition, verbose=args.verbose)
        save(record)
        print(f"      outcome={record['outcome']} decision={record['decision']} "
              f"steps={len(record['steps'])} stop={record['stop_reason']}")

    # Read everything back (saved runs + this session) for the scoreboard.
    records = [json.loads(l) for l in RESULTS_FILE.open(encoding="utf-8")] if RESULTS_FILE.exists() else []
    records = [r for r in records if not args.only or r["scenario"] == args.only]
    scoreboard(records)


if __name__ == "__main__":
    main()
