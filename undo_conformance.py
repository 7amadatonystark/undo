"""
undo_conformance.py — Step I4: don't trust an undo claim; TEST it.

For each action type: build a fresh sandbox, snapshot it, run the action, run its undo, and
compare the world with the snapshot (files and their contents, customer rows, outbox, bank,
inbox). PASS only if they match. A COMPENSABLE action cannot really be undone, so it passes if
its correction exists and points to the original (and nothing else changed). Effects that were
still HELD (Step I2) must be erased completely, so they are tested like reversible ones.

Each action type runs RUNS times with random inputs from a seeded generator (the same inputs
every time). One deliberately BROKEN action (an undo that restores every row but one) shows
that the test catches a bad undo. See UNDO_CONTRACT.md for what passing means.

    python undo_conformance.py            # 100 runs per action type (no API calls)
"""

import random
import string
import sys

from undo import Sandbox
from undo.actions import (ALLOWED_OPS, AddCustomer, DeleteFile, DeleteRows, SendEmail, SetEmail,
                          TransferMoney, WriteFile, read_outbox_line)
from undo.sandbox import COMPANY_CUSTOMERS, COMPANY_FILES

RUNS = 100
SEED = 0


def world(sb):
    """Everything the test compares: snapshot() plus the inbox."""
    return {**sb.snapshot(), "inbox": sb.inbox()}


def word(rng, n=6):
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(n))


# ---------- random inputs, one maker per action type ----------

def random_filter(rng):
    column = rng.choice(["id", "name", "email"])
    op = rng.choice(sorted(ALLOWED_OPS))
    if column == "id":
        value = str(rng.randint(0, 7))
    else:
        name, email = rng.choice(COMPANY_CUSTOMERS)
        value = name if column == "name" else email
    return (column, op, value)


MAKERS = {
    "WriteFile (existing file)": lambda sb, rng: WriteFile(sb, rng.choice(sorted(COMPANY_FILES)),
                                                           word(rng, 20)),
    "WriteFile (new file)": lambda sb, rng: WriteFile(sb, f"{word(rng)}.txt", word(rng, 20)),
    "DeleteFile": lambda sb, rng: DeleteFile(sb, rng.choice(sorted(COMPANY_FILES))),
    "DeleteRows": lambda sb, rng: DeleteRows(sb, rng.choice([None, random_filter(rng)])),
    "AddCustomer": lambda sb, rng: AddCustomer(sb, word(rng).title(), f"{word(rng)}@gmail.com"),
    "SetEmail": lambda sb, rng: SetEmail(sb, random_filter(rng), f"{word(rng)}@gmail.com"),
    "TransferMoney (to savings)": lambda sb, rng: TransferMoney(sb, rng.randint(1, 9000), "savings"),
}
# Compensable: no true undo; a correction must reach the same people.
COMPENSATED = {
    "SendEmail": lambda sb, rng: SendEmail(sb, rng.choice(["all_customers", "boss@company.example",
                                                           COMPANY_CUSTOMERS[0][1]]),
                                           word(rng), word(rng, 30)),
}
# Step I2: an effect cancelled while still held must leave no trace at all.
HELD = {
    "SendEmail (held, erased)": COMPENSATED["SendEmail"],
    "TransferMoney (held, erased)": lambda sb, rng: TransferMoney(sb, rng.randint(1, 9000),
                                                                  f"stranger_{word(rng)}"),
}


class BrokenDeleteRows(DeleteRows):
    """A deliberately BAD undo: it restores every deleted row but one."""
    def undo(self):
        self.saved_rows = self.saved_rows[1:]
        super().undo()


# ---------- the checks ----------

def reversible_passes(make, rng, undo="undo"):
    """Snapshot, do, undo (or erase), compare. Returns True if the world is exactly back."""
    sb = Sandbox(seed="company")
    try:
        before = world(sb)
        action = make(sb, rng)
        action.prepare()
        action.do()
        getattr(action, undo)()
        return world(sb) == before
    finally:
        sb.cleanup()


def compensable_passes(make, rng):
    """Do, undo: the only change left must be the original email plus ONE correction that
    names it and goes to the same address."""
    sb = Sandbox(seed="company")
    try:
        before = world(sb)
        action = make(sb, rng)
        action.prepare()
        action.do()
        action.undo()
        after = world(sb)
        new_lines = after["outbox"][len(before["outbox"]):]
        if {k: v for k, v in after.items() if k != "outbox"} != {k: v for k, v in before.items() if k != "outbox"}:
            return False
        if len(new_lines) != 2:
            return False
        original, correction = (read_outbox_line(line) for line in new_lines)
        return (correction[0] == original[0]
                and correction[1] == f"CORRECTION: please ignore '{original[1]}'")
    finally:
        sb.cleanup()


def rate(check, make, runs, seed):
    rng = random.Random(seed)
    return sum(check(make, rng) for _ in range(runs))


def main():
    sys.stdout.reconfigure(errors="replace")
    runs = int(sys.argv[1]) if len(sys.argv) > 1 else RUNS
    print(f"Undo conformance: {runs} random runs per action type (seed {SEED}).\n")
    print(f"{'ACTION':32s} {'KIND OF TEST':22s} {'PASSED':>12s}")
    rows = [(name, "exact restore", lambda m, r: reversible_passes(m, r), make)
            for name, make in MAKERS.items()]
    rows += [(name, "correction exists", compensable_passes, make)
             for name, make in COMPENSATED.items()]
    rows += [(name, "erased while held", lambda m, r: reversible_passes(m, r, "erase"), make)
             for name, make in HELD.items()]
    rows += [("BrokenDeleteRows (deliberate bug)", "exact restore",
              lambda m, r: reversible_passes(m, r),
              lambda sb, rng: BrokenDeleteRows(sb, rng.choice([None, random_filter(rng)])))]
    for name, test, check, make in rows:
        passed = rate(check, make, runs, SEED)
        print(f"{name:32s} {test:22s} {passed:>6d}/{runs:<5d}")
    print("\nTransferMoney to an outside account and DropTable are IRREVERSIBLE: they have no undo "
          "to test, which is exactly why UNDO gates or holds them.")


if __name__ == "__main__":
    main()
