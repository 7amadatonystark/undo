"""
sweep.py — how do the verifier's thresholds trade CAUGHT against FALSE ALARMS?

For each threshold, try a range of values (keeping the others at their defaults),
run every scenario with UNDO, and count:
    caught       - harmful tasks rolled back      (higher is better)
    false alarms - harmless tasks rolled back     (lower is better)
There is usually no value that is best at both. Finding the sweet spot is the point.

    python sweep.py

Results are saved to sweep.csv: plot caught and false alarms against value,
one chart per threshold.
"""

import csv

from run_benchmark import run_with_undo, score
from scenarios import SCENARIOS
from undo import verifiers

# (verifier, threshold constant in verifiers.py, values to try)
SWEEPS = [
    ("rules", "MASS_CHANGE_LIMIT", [i / 20 for i in range(21)]),     # 0.00 .. 1.00
    ("rules", "SHRINK_LIMIT", [i / 20 for i in range(21)]),          # 0.00 .. 1.00
    ("diff_size", "DIFF_LIMIT", list(range(11))),                    # 0 .. 10
]


def evaluate(verifier_name):
    """Run every scenario once with this verifier; count caught / false alarms."""
    counts = dict(caught=0, missed=0, false_alarm=0, kept=0)
    for sc in SCENARIOS:
        r = run_with_undo(sc, verifier_name, verbose=False)
        counts[score(sc["harmful"], r["decision"])] += 1
    return counts


def main():
    n_harmful = sum(sc["harmful"] for sc in SCENARIOS)
    n_harmless = len(SCENARIOS) - n_harmful
    rows = []

    for verifier_name, constant, values in SWEEPS:
        default = getattr(verifiers, constant)
        print(f"\n{verifier_name}: {constant}  (default {default})")
        print(f"{'VALUE':>7s} {'CAUGHT':>9s} {'FALSE ALARMS':>13s}")
        for value in values:
            setattr(verifiers, constant, value)     # the rules read this at run time
            c = evaluate(verifier_name)
            mark = "  <- default" if value == default else ""
            print(f"{value:>7} {c['caught']:>5d}/{n_harmful:<3d} "
                  f"{c['false_alarm']:>9d}/{n_harmless:<3d}{mark}")
            rows.append(dict(verifier=verifier_name, threshold=constant, value=value,
                             caught=c["caught"], missed=c["missed"],
                             false_alarms=c["false_alarm"], kept=c["kept"],
                             catch_rate=round(c["caught"] / n_harmful, 3),
                             false_alarm_rate=round(c["false_alarm"] / n_harmless, 3)))
        setattr(verifiers, constant, default)       # put it back for the next sweep

    with open("sweep.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    print("\nSaved sweep.csv")


if __name__ == "__main__":
    main()
