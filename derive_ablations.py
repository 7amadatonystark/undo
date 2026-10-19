"""
derive_ablations.py — Step 6: three ablations computed by REPLAY instead of new runs.

"No verifier", "no rollback" and "no hold" change only what UNDO does AFTER the agent has
finished: the agent sees exactly the same tool results either way (the gate is still on, and
a held email still says "done"). So for every saved full-UNDO run, the agent's exact tool calls
are run again in a fresh fake world under each ablation, the ablation's decision is applied,
and the result is scored with the answer key. No API calls are needed, and each ablation is
compared with full UNDO on exactly the same agent work (a paired comparison).

A replay must reproduce every saved tool result; if it doesn't, that run is skipped (and
reported), never guessed. Each derived record is saved to results/runs.jsonl with
method="replay" and derived_from=<the full-UNDO run>.

    python derive_ablations.py
"""

import sys
import time
from datetime import datetime
from types import SimpleNamespace

from agent import call_tool
from run_benchmark import BLOCKED_REASON, NO_VERIFIER_REASON, check_done, outcome_label, settings, verdict
from run_final import REPLAYED, all_scenarios, final_records, key, save_shared
from undo import Sandbox, UndoRuntime


def run_id(r):
    return f"{r['batch']}|{r['scenario']}|{r['condition']}|{r['run']}|{r['timestamp']}"


def derive(scenario, original, condition):
    """One ablation of one full-UNDO run, by replay. Returns the record, or None if the replay
    did not reproduce the saved tool results."""
    s = settings(condition)
    sb = Sandbox(seed="company")
    try:
        if scenario.get("setup"):
            scenario["setup"](sb)
        rt = UndoRuntime(approver=scenario["approver"], verbose=False, gate=s["gate"],
                         per_call=s["per_call"], hold_seconds=s["hold"], seed=original["run"])
        for n, step in enumerate(original["steps"]):
            call = SimpleNamespace(id=f"replay_{n}",
                                   function=SimpleNamespace(name=step["tool"], arguments=step["args"]))
            if call_tool(call, sb, rt) != step["result"]:
                return None
            rt.tick()
        changes = [c["text"] for c in rt.change_summary(sb)]
        if changes != original["changes"]:
            return None

        # The decision: the same verifier answer, because the verifier would have seen the very
        # same task and changes. Only "no verifier" decides differently.
        blocked = BLOCKED_REASON in original["reasons"]
        if condition == "no_verifier" and not blocked:
            decision, reasons, judgments = "commit", [NO_VERIFIER_REASON], None
        else:
            decision, reasons = original["decision"], original["reasons"]
            judgments = original.get("judgments")
        rolled_back = decision == "rollback" and s["rollback"]
        t0 = time.perf_counter()
        rt.rollback() if rolled_back else rt.commit()
        overhead = (time.perf_counter() - t0) * 1000
        return dict(original, timestamp=datetime.now().isoformat(timespec="seconds"),
                    condition=condition, method="replay", derived_from=run_id(original),
                    judgments=judgments, decision=decision, reasons=reasons,
                    rolled_back=rolled_back,
                    verdict=verdict(original["damaged_before"], decision, blocked),
                    damaged=scenario["damaged"](sb), done=check_done(scenario, sb),
                    outcome=outcome_label(scenario, sb), hold_seconds=rt.hold_seconds,
                    holds=rt.hold_report(), approvals_asked=rt.approvals_asked,
                    overhead_ms=round(original["overhead_ms"] + overhead, 3),
                    report=None, report_looks=[], honesty=None)   # the report belongs to the original
    finally:
        sb.cleanup()


def main():
    sys.stdout.reconfigure(errors="replace")
    by_name = {s["name"]: s for s in all_scenarios()}
    records = final_records()
    done = {k for k, r in records.items() if r.get("method") == "replay"}
    originals = [r for r in records.values() if r["condition"] == "undo"
                 and r.get("language", "en") == "en" and r["scenario"] in by_name]
    made, skipped = 0, 0
    for original in originals:
        for condition in REPLAYED:
            if key(original["scenario"], condition, "en", original["run"]) in done:
                continue
            record = derive(by_name[original["scenario"]], original, condition)
            if record is None:
                skipped += 1
                continue
            save_shared(record)
            made += 1
    print(f"Derived {made} ablation run(s) from {len(originals)} full-UNDO runs; "
          f"{skipped} skipped because the replay did not match.")


if __name__ == "__main__":
    main()
