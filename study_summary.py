"""
study_summary.py — Step 11: compare the two conditions of the approval-fatigue study.

    python study_summary.py             # study_results.csv (real, IRB-approved data)
    python study_summary.py --practice  # study_practice.csv

For each condition: participants, decisions per person, the share of harmful actions they
approved, and the time per decision. Fisher's exact test (from analyze.py) checks whether the
difference in harmful approvals could be luck. With few participants, treat it as a pilot.
"""

import csv
import statistics
import sys
from pathlib import Path

from analyze import fisher_exact, rate_line

HERE = Path(__file__).parent


def main():
    path = HERE / ("study_practice.csv" if "--practice" in sys.argv else "study_results.csv")
    if not path.exists():
        raise SystemExit(f"No data yet in {path.name}.")
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"{path.name}: {len(rows)} participant(s)\n")
    counts = {}
    for condition, meaning in (("A", "asked about EVERY action"),
                               ("B", "UNDO: asked only about irreversible actions")):
        mine = [r for r in rows if r["condition"] == condition]
        if not mine:
            continue
        shown = sum(int(r["harmful_shown"]) for r in mine)
        approved = sum(int(r["harmful_approved"]) for r in mine)
        counts[condition] = (approved, shown)
        print(f"Condition {condition} ({meaning}): {len(mine)} participant(s)")
        print(f"  decisions per person:    {statistics.mean(int(r['decisions']) for r in mine):.1f}")
        print(f"  harmful actions approved: {rate_line(approved, shown)}")
        print(f"  median seconds/decision:  "
              f"{statistics.median(float(r['median_seconds_per_decision']) for r in mine):.1f}")
    if len(counts) == 2:
        (a, n_a), (b, n_b) = counts["A"], counts["B"]
        p = fisher_exact(a, n_a - a, b, n_b - b)
        print(f"\nFisher's exact test, harmful approvals A vs. B: p = {p:.2g} "
              f"(p < 0.05: unlikely to be luck). Each participant sees several harmful actions, "
              f"so these are not fully independent: treat p as a rough guide.")


if __name__ == "__main__":
    main()
