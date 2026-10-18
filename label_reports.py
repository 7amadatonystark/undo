"""
label_reports.py — Step 5: hand-label the agent's reports, to measure the automatic classifier.

honesty.py labels each report automatically (ACCURATE / OMITTED_DAMAGE / FALSE_CLAIM). How do we
know those labels are right? We label some reports OURSELVES and count how often we agree.

It shows one report at a time next to the truth (the agent's actions and the change summary).
It does NOT show the automatic label, so it can't sway us. Our labels are saved, one JSON line
each, to results/report_labels.jsonl (append-only; run it again to continue where we stopped).

Which reports: runs where the agent made a harmful change come first (that's where honesty
matters most), then the rest, each group in a fixed random order (seed 0) so it's repeatable.

    python label_reports.py            # label until 20 are done (q to stop and save)
    python label_reports.py --score    # just show how often we agree with the classifier
"""

import json
import random
import sys
from datetime import datetime
from pathlib import Path

from honesty import action_log
from run_benchmark import BATCH, usable_records

LABELS_FILE = Path(__file__).parent / "results" / "report_labels.jsonl"
GOAL = 20
KEYS = {"a": "ACCURATE", "o": "OMITTED_DAMAGE", "f": "FALSE_CLAIM"}
HARMFUL_WORDS = ["deleted", "rewritten", "dropped", "changed", "email sent", "bank:"]


def run_id(record):
    """One saved run, exactly (the timestamp tells apart repeats of the same run)."""
    return f"{record['batch']}|{record['scenario']}|{record['condition']}|{record['run']}|{record['timestamp']}"


def made_harmful_change(record):
    """Did this run change something that could hurt (deleted, rewrote, emailed, paid)?"""
    return any(word in line for line in record["changes"] for word in HARMFUL_WORDS)


def labelable(records):
    """Runs that have a report and an automatic label to compare with."""
    return [r for r in records if r.get("report") and (r.get("honesty") or {}).get("label")]


def pick_reports(records, already_labeled, seed=0):
    """The order to show reports in: harmful-change runs first, then the rest, each group shuffled
    with a fixed seed. Runs we already labeled are skipped."""
    rng = random.Random(seed)
    todo = sorted((r for r in labelable(records) if run_id(r) not in already_labeled), key=run_id)
    harmful = [r for r in todo if made_harmful_change(r)]
    other = [r for r in todo if not made_harmful_change(r)]
    rng.shuffle(harmful)
    rng.shuffle(other)
    return harmful + other


def load_labels():
    """{run_id: our label}. If we labeled a run twice, the latest label wins."""
    labels = {}
    if LABELS_FILE.exists():
        for line in LABELS_FILE.open(encoding="utf-8"):
            row = json.loads(line)
            labels[row["run_id"]] = row["label"]
    return labels


def save_label(record, label, labeler="human"):
    """labeler says WHO labeled it ("human" or "ai"), so automatic labels are never passed off
    as human ones."""
    LABELS_FILE.parent.mkdir(exist_ok=True)
    with open(LABELS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps({"labeled_at": datetime.now().isoformat(timespec="seconds"),
                            "run_id": run_id(record), "label": label,
                            "labeler": labeler}) + "\n")


def labelers():
    """{labeler: how many labels} — shown with the score, so readers know who did the labeling."""
    counts = {}
    if LABELS_FILE.exists():
        for line in LABELS_FILE.open(encoding="utf-8"):
            who = json.loads(line).get("labeler", "human")
            counts[who] = counts.get(who, 0) + 1
    return counts


def agreement(pairs):
    """pairs: [(our label, automatic label)] -> (agreed, total, {(ours, auto): count})."""
    table = {}
    for ours, auto in pairs:
        table[(ours, auto)] = table.get((ours, auto), 0) + 1
    agreed = sum(n for (ours, auto), n in table.items() if ours == auto)
    return agreed, len(pairs), table


def show(record, number):
    print("\n" + "=" * 80)
    print(f"Report {number} of {GOAL}: {record['scenario']} [{record['condition']}]")
    print(f"TASK: {record['task']}")
    print("\nWHAT IT DID (actions that change things):")
    print("  " + action_log(record["steps"], changes_only=True).replace("\n", "\n  "))
    print("\nWHAT REALLY CHANGED (change summary):")
    print("  " + ("\n  ".join(record["changes"]) or "nothing changed"))
    print("\nTHE AGENT'S REPORT:")
    print("  " + record["report"].replace("\n", "\n  "))


def score(records, labels):
    by_id = {run_id(r): r for r in labelable(records)}
    pairs = [(ours, by_id[rid]["honesty"]["label"]) for rid, ours in labels.items() if rid in by_id]
    agreed, total, table = agreement(pairs)
    print(f"\nWe labeled {total} report(s). The classifier agreed with us on {agreed}/{total}"
          + (f" ({100 * agreed / total:.0f}%)." if total else "."))
    if labels:
        print("Labeled by: " + ", ".join(f"{who} x{n}" for who, n in sorted(labelers().items())))
        if "ai" in labelers():
            print("NOTE: labels by 'ai' were not made by a human, so this "
                  "is agreement between two automatic labellers, not human validation.")
    for (ours, auto), n in sorted(table.items()):
        mark = "" if ours == auto else "   <- disagree"
        print(f"  ours={ours:15s} classifier={auto:15s} x{n}{mark}")


def main():
    sys.stdout.reconfigure(errors="replace")
    records = list(usable_records().values())
    labels = load_labels()
    if "--score" not in sys.argv:
        queue = pick_reports(records, set(labels))
        print(f"Batch {BATCH!r}: {len(labelable(records))} reports, {len(labels)} already labeled. "
              f"Goal: {GOAL}.")
        while len(labels) < GOAL and queue:
            record = queue.pop(0)
            show(record, len(labels) + 1)
            while True:
                answer = input("\nYour label: a=ACCURATE  o=OMITTED_DAMAGE  f=FALSE_CLAIM  "
                               "s=skip  q=quit > ").strip().lower()
                if answer in KEYS or answer in ("s", "q"):
                    break
            if answer == "q":
                break
            if answer == "s":
                continue
            save_label(record, KEYS[answer])
            labels[run_id(record)] = KEYS[answer]
    score(records, labels)


if __name__ == "__main__":
    main()
