"""
overhead_scaling.py — Step L5: how does UNDO's overhead grow with the size of the world?

For worlds with 10, 100, 1,000 and 10,000 extra files and customer rows (fake data, seed 0,
files of about 2.7 KB), a SCRIPTED task (no API calls, so the timing is fair) is run REPEATS
times, and the median time of each part is reported:
  snapshot   - one full snapshot() of the world (the old way needed two per task)
  prepare    - saving undo information, summed over the task's actions
  rollback   - undoing the whole task
  task_full  - UNDO's whole overhead the OLD way: snapshot before + after, summary, rollback
  task_journal - the same task the NEW way: the change journal (Step L5) instead of snapshots
  dry_run    - one raw-tool dry run (Step L4), which still copies the world
The world is rolled back after every repeat, so every repeat starts from the same state.

    python overhead_scaling.py               # 20 repeats per size (a few minutes)
    python overhead_scaling.py --quick       # 3 repeats, sizes up to 1,000 (for testing)
"""

import json
import statistics
import sys
import time
from datetime import date
from pathlib import Path

from analyze import FIGURES
from scenarios import human_says_yes
from undo import Sandbox, UndoRuntime
from undo.actions import DeleteFile, DeleteRows, SendEmail, WriteFile
from undo.changes import summarize
from undo.classifier import dry_run
from undo.raw_tools import RawCommand

SIZES = [10, 100, 1000, 10000]
REPEATS = 20
DRY_RUN_REPEATS = 3


def task(sb):
    """The scripted task: a little of everything, the same at every size."""
    return [WriteFile(sb, "report.txt", "new numbers\n"), DeleteFile(sb, "cache.tmp"),
            DeleteRows(sb, ("email", "=", "test@test.com")),
            SendEmail(sb, "boss@company.example", "Done", "All tidy.")]


def ms(seconds):
    return seconds * 1000


def one_repeat(sb):
    """Time every part once. Returns {part: milliseconds}."""
    times = {}
    t0 = time.perf_counter()
    sb.snapshot()
    times["snapshot"] = ms(time.perf_counter() - t0)

    actions = task(sb)
    t0 = time.perf_counter()
    for action in actions:
        action.prepare()
    times["prepare"] = ms(time.perf_counter() - t0)

    # Both ways run the task through the same UndoRuntime (same journal, same rollback); the
    # only difference is how the change summary is made.
    for way in ("task_full", "task_journal"):
        rt = UndoRuntime(approver=human_says_yes, verbose=False)
        t0 = time.perf_counter()
        before = sb.snapshot() if way == "task_full" else None
        for action in task(sb):
            rt.execute(action)
        if way == "task_full":
            summarize(before, sb.snapshot())        # the OLD way: two full snapshots
        else:
            rt.change_summary(sb)                   # the NEW way: the change journal
        t1 = time.perf_counter()
        rt.rollback()
        times["rollback"] = ms(time.perf_counter() - t1)
        times[way] = ms(time.perf_counter() - t0)
    return times


def dry_run_time(sb):
    t0 = time.perf_counter()
    dry_run(RawCommand(sb, "run_shell", "rm cache.tmp"))
    return ms(time.perf_counter() - t0)


def measure(sizes, repeats):
    results = {}
    for size in sizes:
        sb = Sandbox(seed="company", size=size)
        try:
            runs = [one_repeat(sb) for _ in range(repeats)]
            results[size] = {part: statistics.median(r[part] for r in runs) for part in runs[0]}
            # A dry run copies the whole world, so it is slow on big worlds: fewer repeats.
            results[size]["dry_run"] = statistics.median(dry_run_time(sb) for _ in range(DRY_RUN_REPEATS))
        finally:
            sb.cleanup()
        print(f"  size {size:>6d}: " + "  ".join(f"{k} {v:8.1f} ms" for k, v in results[size].items()),
              flush=True)
    return results


def graph(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sizes = list(results)
    fig, ax = plt.subplots(figsize=(7, 5))
    for part, label, color in (("task_full", "full snapshots (before)", "#c0392b"),
                               ("task_journal", "change journal (after)", "#27ae60"),
                               ("dry_run", "raw-tool dry run", "#7f8c8d")):
        ax.plot(sizes, [results[s][part] for s in sizes], "o-", label=label, color=color)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("world size (extra files and customer rows)")
    ax.set_ylabel("UNDO overhead per task (ms, median)")
    ax.set_title("Overhead against world size (log-log)")
    ax.legend()
    fig.tight_layout()
    FIGURES.mkdir(exist_ok=True)
    fig.savefig(FIGURES / "overhead_scaling.png", dpi=150)
    plt.close(fig)
    print(f"Saved {FIGURES / 'overhead_scaling.png'}")


def slope(results, part):
    """The slope on log-log axes between the smallest and largest world: about 1 means the
    time grows in proportion to the world (10x bigger -> 10x slower); about 0 means flat."""
    import math
    small, big = min(results), max(results)
    return math.log(results[big][part] / results[small][part]) / math.log(big / small)


def main():
    sys.stdout.reconfigure(errors="replace")
    quick = "--quick" in sys.argv
    sizes, repeats = (SIZES[:3], 3) if quick else (SIZES, REPEATS)
    print(f"Timing a scripted task, {repeats} repeats per world size (medians):")
    results = measure(sizes, repeats)
    for part in ("task_full", "task_journal", "dry_run"):
        print(f"  log-log slope of {part}: {slope(results, part):.2f}")
    print("A slope near 1 means the overhead grows in proportion to the world; near 0, it stays "
          "flat however big the world gets.")
    graph(results)
    out = Path(__file__).parent / "results" / "overhead_scaling.json"
    out.write_text(json.dumps({"date": date.today().isoformat(), "repeats": repeats,
                               "median_ms": results}, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
