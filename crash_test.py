"""
crash_test.py — does UNDO's recovery survive a crash mid-task or mid-rollback? (Step 7, no API)

Each trial:
  1. Build a fresh sandbox and remember its starting state.
  2. In a CHILD process, run a scripted task (reversible actions only) through a journaling
     UndoRuntime, and kill the process with os._exit(1) at a seeded-random moment — either
     during the task, or during a rollback of it.
  3. Back in the parent, call recover() (twice, to prove it is safe to repeat).
  4. Check the sandbox is byte-for-byte back at its starting state.

Reversible actions only: a sent email or a money transfer cannot return to the exact starting
state (that is what "compensable" and "irreversible" mean), so they can't be in a perfect-match
test. Recovery still treats them exactly as rollback does.

    python crash_test.py             # 100 trials
    python crash_test.py 500         # a different number of trials

The crash lands on a journal write, which brackets every do() and undo(). So a trial can crash
before an action runs, after it ran but before it was marked done, or part-way through a rollback.
"""

import os
import random
import subprocess
import sys
import tempfile

from undo import Sandbox, UndoRuntime, recover
import undo.runtime as runtime_module
from undo.actions import WriteFile, DeleteFile, DeleteRows, AddCustomer, SetEmail


def human_says_no(action):
    return False


# Scripted tasks: each returns the actions to run, in order. All reversible.
def task_edit_files(sb):
    return [WriteFile(sb, "report.txt", "edited report\n"),
            DeleteFile(sb, "old_notes.txt"),
            WriteFile(sb, "summary.txt", "a brand new file\n")]


def task_edit_customers(sb):
    return [DeleteRows(sb, ("email", "=", "test@test.com")),
            AddCustomer(sb, "Laila Adel", "laila.adel@gmail.com"),
            SetEmail(sb, ("name", "=", "Karim Fawzy"), "karim.new@gmail.com")]


def task_mixed(sb):
    return [WriteFile(sb, "config.json", '{"mode": "debug"}\n'),
            DeleteRows(sb, ("id", ">", 4)),
            AddCustomer(sb, "Nour Amin", "nour.amin@outlook.com"),
            DeleteFile(sb, "cache.tmp")]


TASKS = [task_edit_files, task_edit_customers, task_mixed]


# ---------------- the child process: run a task and crash ----------------

def run_worker(root, seed):
    """Attach to the sandbox at `root`, run a seeded task, and crash partway through."""
    rng = random.Random(seed)
    sb = Sandbox(root=root)
    rt = UndoRuntime(approver=human_says_no, verbose=False)
    task = rng.choice(TASKS)
    mode = rng.choice(["during_task", "during_rollback"])
    actions = task(sb)

    # Crash on the Nth journal write, for a seeded N. If N is past the end, we still crash at the
    # very end (below), so a crash always happens.
    budget = 2 * len(actions) + 2 + 2 * len(actions) + 2
    crash_at = rng.randint(1, budget)
    counter = {"n": 0}
    real_append = runtime_module.append_journal

    def crashing_append(path, record):
        counter["n"] += 1
        if counter["n"] >= crash_at:
            os._exit(1)                 # pull the plug right before this record reaches disk
        real_append(path, record)

    runtime_module.append_journal = crashing_append

    for action in actions:
        rt.execute(action)
    if mode == "during_rollback":
        rt.rollback()
    os._exit(1)                         # if we never hit crash_at, crash now (still mid-task)


# ---------------- the parent process: recover and check ----------------

def run_trials(n):
    perfect = 0
    failures = []
    for i in range(n):
        seed = i
        sb = Sandbox(seed="company")                 # a fresh, seeded world (its own temp dir)
        start = sb.snapshot()
        try:
            subprocess.run([sys.executable, __file__, "--worker", sb.root, str(seed)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            recover(sb)                              # finish what the crash interrupted
            recover(sb)                              # run again: must be safe and change nothing
            if sb.snapshot() == start:
                perfect += 1
            else:
                failures.append(seed)
        finally:
            sb.cleanup()
    return perfect, failures


def main():
    if len(sys.argv) >= 3 and sys.argv[1] == "--worker":
        run_worker(sys.argv[2], int(sys.argv[3]))
        return                                       # run_worker never returns (it os._exit's)

    trials = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    print(f"Running {trials} crash-and-recover trials (no API)...")
    perfect, failures = run_trials(trials)
    print(f"\nRecovered perfectly: {perfect}/{trials}")
    if failures:
        print(f"Did NOT return to the starting state on these seeds: {failures}")
    else:
        print("Every crash recovered to the exact starting state.")


if __name__ == "__main__":
    main()
