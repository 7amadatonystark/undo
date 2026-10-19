"""
hold_experiment.py — Step I2: does a short delay turn irreversible actions into reversible ones?

For each hold time (HOLD_TIMES), every saved full-UNDO run is REPLAYED: the agent's exact tool
calls are run again in a fresh fake world through UndoRuntime with that hold window, on the
simulated clock (one step = STEP_SECONDS), and UNDO's saved decision (commit or roll back) is
applied at the end. No API calls are needed, because the hold cannot change what the agent did
or what the verifier decided: the agent is told "done" either way, and the change summary the
verifier reads is the same.

Reported per hold time:
  - damage rate at the end (with a 95% Wilson interval)
  - harmful effects cancelled while still held (they never left)
  - the extra delay a hold adds to harmless tasks (released only after the check)
and figures/hold_window.png: damage against hold time.

    python hold_experiment.py              # the final data (results/runs.jsonl)
"""

import statistics
import sys
from types import SimpleNamespace

from agent import call_tool
from analyze import FIGURES, wilson
from injection_scenarios import INJECTION_SCENARIOS
from run_final import final_records
from heldout_scenarios import HELDOUT_SCENARIOS
from scenarios import PAIR_SCENARIOS, SCENARIOS
from undo import Sandbox, UndoRuntime

HOLD_TIMES = [0, 10, 60, 300]


def replay(scenario, record, hold):
    """One saved run, replayed with this hold time. Returns a result dict, or None if the replay
    did not reproduce the saved tool results (then it is not a fair replay and is skipped)."""
    sb = Sandbox(seed="company")
    try:
        if scenario.get("setup"):
            scenario["setup"](sb)
        rt = UndoRuntime(approver=scenario["approver"], verbose=False, hold_seconds=hold,
                         seed=record["run"])
        for n, step in enumerate(record["steps"]):
            call = SimpleNamespace(id=f"replay_{n}",
                                   function=SimpleNamespace(name=step["tool"], arguments=step["args"]))
            if call_tool(call, sb, rt) != step["result"]:
                return None
            rt.tick()
        rt.rollback() if record["decision"] == "rollback" else rt.commit()
        holds = rt.hold_report()
        return dict(damaged=bool(scenario["damaged"](sb)), harmful=record["damaged_before"],
                    decision=record["decision"],
                    cancelled=sum(h["outcome"] == "cancelled while held" for h in holds),
                    delays=[h["delivered_at"] - h["sent_at"] for h in holds
                            if h.get("delivered_at") is not None])
    finally:
        sb.cleanup()


def run(records):
    by_name = {s["name"]: s for s in SCENARIOS + INJECTION_SCENARIOS + PAIR_SCENARIOS
               + HELDOUT_SCENARIOS}
    results, skipped = {}, 0
    for hold in HOLD_TIMES:
        results[hold] = []
        for r in records:
            out = replay(by_name[r["scenario"]], r, hold)
            if out is None:
                skipped += 1
            else:
                results[hold].append(out)
    return results, skipped // len(HOLD_TIMES)


def report(results, skipped):
    print(f"\n{'HOLD':>6s} {'DAMAGE LEFT':>22s} {'95% CI':>13s} {'HARMFUL CANCELLED':>18s} "
          f"{'EXTRA DELAY (harmless)':>23s}")
    for hold, rs in results.items():
        k, n = sum(r["damaged"] for r in rs), len(rs)
        low, high = wilson(k, n)
        cancelled = sum(r["cancelled"] for r in rs if r["harmful"])
        delays = [d for r in rs if not r["harmful"] and r["decision"] == "commit" for d in r["delays"]]
        delay = f"{statistics.mean(delays):.0f} s avg" if delays else "-"
        print(f"{hold:>5d}s {k:>10d}/{n:<4d} ({100 * k / n:4.1f}%) {100 * low:5.1f}-{100 * high:4.1f}% "
              f"{cancelled:>18d} {delay:>23s}")
    if skipped:
        print(f"({skipped} run(s) skipped: their replay did not reproduce the saved tool results)")
    print("\nA longer hold can cancel more harmful effects before they leave, but makes harmless "
          "emails and payments wait longer.")


def graph(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    holds = list(results)
    rates, errors = [], [[], []]
    for hold in holds:
        k, n = sum(r["damaged"] for r in results[hold]), len(results[hold])
        low, high = wilson(k, n)
        rate = 100 * k / n
        rates.append(rate)
        errors[0].append(rate - 100 * low)
        errors[1].append(100 * high - rate)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar([str(h) for h in holds], rates, yerr=errors, fmt="o-", capsize=5, color="#2c7fb8")
    ax.set_xlabel("hold window (simulated seconds)")
    ax.set_ylabel("runs with damage left (%)")
    ax.set_title("Damage against hold time (full UNDO, 95% Wilson CI)")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "hold_window.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIGURES / 'hold_window.png'}")


def main():
    sys.stdout.reconfigure(errors="replace")
    records = [r for r in final_records().values()
               if r["condition"] == "undo" and r.get("language", "en") == "en"]
    print(f"Replaying {len(records)} full-UNDO runs at hold times {HOLD_TIMES} (no API calls).")
    results, skipped = run(records)
    report(results, skipped)
    graph(results)


if __name__ == "__main__":
    main()
