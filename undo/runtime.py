"""
runtime.py — the heart of UNDO.

For every action the agent attempts:
  1. CLASSIFY  - the kind of THIS call (undo/classifier.py, Step I1), not just the tool's
  2. GATE      - irreversible actions need a human "yes" first
  3. STAGE     - prepare() saves undo info, then do() runs it
  4. COMMIT or ROLLBACK at the end:
       - commit()   keeps everything
       - rollback() undoes every completed action in REVERSE order

Crash recovery (Step 7): a WRITE-AHEAD JOURNAL records each step on disk BEFORE it happens, so
if the program is killed mid-task or mid-rollback, recover() can finish the job on restart.
  - before an action's do(): append {do_begin, seq, action} and fsync; after it: {do_done, seq}
  - rollback writes {rollback_begin}, then {undo_begin/undo_done, seq} per action, then {rollback_done}
  - commit writes {commit}
recover(sandbox) reads the journal: if the task never committed, it undoes every reversible
action that ran and is not already undone, in reverse order. It is safe to run more than once.

Hold window (Step I2): effects that LEAVE the sandbox (emails, money transfers) can be held for
hold_seconds of SIMULATED time, like Gmail's "Undo send". While held they never left, so a
rollback erases them completely. commit() releases them; a hold that runs out during the task
releases them unchecked. Time is a simulated clock: every agent step moves it on by
STEP_SECONDS, so runs stay fast. The outside world is modelled simply: a released email is READ
after a random delay (seeded), and a correction sent after that does not count as undoing it.

It also records numbers for the benchmark: how many approvals were asked, and how much time
UNDO added on top of the actions themselves (now including the journal's fsyncs).
"""

import json
import os
import random
import time

from .actions import (IRREVERSIBLE, AddCustomer, DeleteFile, DeleteRows, DropTable, SendEmail,
                      SetEmail, TransferMoney, WriteFile, from_journal, to_journal)
from .changes import summarize
from .classifier import classify, classify_effects, default_kind

# ---------- Step I2: the simulated clock ----------
from .settings import HOLD_SECONDS, READ_DELAY_MEAN, STEP_SECONDS   # undo/settings.py


class Blocked(Exception):
    """Raised when a human (or policy) refuses an irreversible action."""


def append_journal(path, record):
    """Append one record as a JSON line and force it to disk, so a crash right after can't lose
    it. fsync is what makes the journal trustworthy (and what makes UNDO's overhead realistic)."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())


def read_journal(path):
    """The journal's records in order. A crash can leave a half-written final line; that line is
    incomplete, so we drop any line that doesn't parse."""
    records = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except ValueError:
                    pass                    # a torn last line from a crash mid-write; ignore it
    return records


class UndoRuntime:
    def __init__(self, approver, verbose=True, enabled=True, gate=True, per_call=True,
                 hold_seconds=0, seed=0, ask_every=False):
        # approver(action) -> True/False. In the demo it asks you at the keyboard;
        # in the benchmark it is a function that plays the role of the human.
        # enabled=False is the "without UNDO" baseline: no human gate on irreversible
        # actions, so every action just runs. (rollback still works if called, but the
        # benchmark never calls it in this mode.)
        # gate=False is a Step 6 ablation: UNDO still stages and can roll back, but irreversible
        # actions are NOT paused for a human. It shows how much the gate alone contributes.
        # per_call=False is another ablation: every call gets its TOOL's default kind, as if
        # there were no per-call classifier (Step I1).
        self.approver = approver
        self.gate = gate
        self.per_call = per_call
        # ask_every=True is Step I3's "permission only" baseline: EVERY action that changes
        # something waits for the approver, not just irreversible ones.
        self.ask_every = ask_every
        self.kinds = []             # per action: tool, default kind, this call's kind, reasons
        # Step I2: the hold window and the simulated clock
        self.hold_seconds = hold_seconds
        self.clock = 0.0            # simulated seconds since the task started
        self.holds = []             # one record per effect that leaves the sandbox
        self.held = []              # the records still held right now
        self.rng = random.Random(seed)      # when released emails get read (fixed seed)
        # Step L5: the change journal. The state of each thing an action touches, saved just
        # before the FIRST action touches it, so the change summary needs no full snapshot.
        self.before_images = {}
        self.verbose = verbose
        self.enabled = enabled
        self.completed = []         # stack of actions that actually ran
        self.approvals_asked = 0
        self.overhead_seconds = 0.0  # time spent in UNDO's own bookkeeping
        self.actions_seen = 0
        self.journal_path = None    # learned from the first action's sandbox
        self._seq = 0               # a number for each action, so the journal can name it
        self._seqs = []             # the seq of each action in self.completed (same order)

    def log(self, msg):
        if self.verbose:
            print(msg)

    def journal(self, record):
        """Write one journal record, if we know where the journal is. Counted as overhead."""
        if self.journal_path:
            t0 = time.perf_counter()
            append_journal(self.journal_path, record)
            self.overhead_seconds += time.perf_counter() - t0

    def execute(self, action):
        self.actions_seen += 1
        self.journal_path = self.journal_path or action.sb.journal_path
        t0 = time.perf_counter()

        self.remember_before(action)            # Step L5: before-images for the change summary

        # 1. classify THIS call. The result is stored on the action, so the gate, the journal,
        # rollback and recover() all use the same kind.
        tool_kind = default_kind(action)
        kind, reasons = classify(action, action.sb) if self.per_call else (tool_kind, [])
        action.kind = kind
        self.kinds.append({"action": type(action).__name__, "default": tool_kind,
                           "kind": kind, "reasons": reasons})

        # 2. gate. The gate only runs when UNDO is enabled;
        # without UNDO, irreversible actions are not paused for a human.
        if (action.kind == IRREVERSIBLE or self.ask_every) and self.enabled and self.gate:
            self.approvals_asked += 1
            self.log(f"[UNDO] Agent wants: {action.describe():40s} -> {action.kind}  paused")
            self.overhead_seconds += time.perf_counter() - t0
            if not self.approver(action):
                self.log("[UNDO] Blocked by human.")
                raise Blocked(action.describe())
            t0 = time.perf_counter()

        # 3. stage: save undo information
        action.prepare()
        self.overhead_seconds += time.perf_counter() - t0

        # Write-ahead: record the action (with its saved undo info) BEFORE doing it, so a crash
        # during do() still leaves a record recover() can undo.
        self._seq += 1
        seq = self._seq
        self.journal({"event": "do_begin", "seq": seq, "action": to_journal(action)})

        try:
            action.do()             # the real effect (not counted as overhead)
        except Exception:
            # The action may have half-happened. Put it on the stack anyway so
            # rollback() tries to undo it, then let the caller see the crash.
            self.completed.append(action)
            self._seqs.append(seq)
            self.log(f"[UNDO] Agent wants: {action.describe():40s} -> CRASHED")
            raise

        self.completed.append(action)
        self._seqs.append(seq)
        self.journal({"event": "do_done", "seq": seq})
        if hasattr(action, "real_changes"):            # a raw command (Step L4): did the dry
            self.kinds[-1]["real_kind"] = classify_effects(action.real_changes(), action.sb)[0]
        if self.enabled and isinstance(action, (SendEmail, TransferMoney)):
            self.hold(action)
        if action.kind != IRREVERSIBLE:
            self.log(f"[UNDO] Agent wants: {action.describe():40s} -> {action.kind}  ran")

    # ---------- Step L5: the change journal ----------

    @staticmethod
    def touches(action):
        """The things an action can change: ("file", name), ("customers",), ("outbox",),
        ("bank",), or ("everything",) for a raw command (nobody knows in advance)."""
        if isinstance(action, (WriteFile, DeleteFile)):
            return [("file", action.name)]
        if isinstance(action, (DeleteRows, AddCustomer, SetEmail, DropTable)):
            return [("customers",)]
        if isinstance(action, SendEmail):
            return [("outbox",)]
        if isinstance(action, TransferMoney):
            return [("bank",)]
        return [("everything",)]

    @staticmethod
    def capture(sb, thing):
        """The current state of one thing."""
        return {"file": lambda: sb.read_file(thing[-1]), "customers": sb.customer_rows,
                "outbox": sb.outbox_lines, "bank": sb.bank,
                "everything": sb.snapshot}[thing[0]]()

    def remember_before(self, action):
        for thing in self.touches(action):
            if thing not in self.before_images:
                self.before_images[thing] = self.capture(action.sb, thing)

    def change_summary(self, sb):
        """What changed since the task started, from the journal: only the things actions
        touched are read again, so the cost grows with the work, not with the size of the
        world. A raw command touched "everything", so then a full snapshot is compared."""
        images = self.before_images
        if ("everything",) in images:
            before, after = dict(images[("everything",)]), sb.snapshot()
            before["files"] = dict(before["files"])
            for thing, state in images.items():     # things touched BEFORE the raw command
                if thing == ("everything",):
                    break                           # (dicts keep the order things were added)
                if thing[0] == "file":
                    if state is None:
                        before["files"].pop(thing[1], None)
                    else:
                        before["files"][thing[1]] = state
                elif thing[0] != "everything":
                    before[thing[0]] = state
            return summarize(before, after)
        names = [t[1] for t in images if t[0] == "file"]
        before = {"files": {n: images[("file", n)] for n in names if images[("file", n)] is not None},
                  "customers": images.get(("customers",)), "outbox": images.get(("outbox",), []),
                  "bank": images.get(("bank",), {})}
        after = {"files": {n: sb.read_file(n) for n in names if sb.read_file(n) is not None},
                 "customers": sb.customer_rows() if ("customers",) in images else None,
                 "outbox": sb.outbox_lines() if ("outbox",) in images else [],
                 "bank": sb.bank() if ("bank",) in images else {}}
        return summarize(before, after)

    # ---------- Step I2: hold, release, erase ----------

    def hold(self, action):
        """An effect that leaves the sandbox: hold it, or deliver it now if there is no hold."""
        record = {"action": action, "sent_at": self.clock,
                  "release_at": self.clock + self.hold_seconds}
        self.holds.append(record)
        if self.hold_seconds > 0:
            self.held.append(record)
            self.log(f"       held until t={record['release_at']:g}s: {action.describe()}")
        else:
            self.deliver(record, "sent at once")

    def deliver(self, record, how):
        """It really leaves now. From here on only its normal undo is possible."""
        if record in self.held:
            self.held.remove(record)
        record.update(outcome=how, delivered_at=self.clock)
        if isinstance(record["action"], SendEmail):
            record["read_at"] = self.clock + self.rng.expovariate(1 / READ_DELAY_MEAN)

    def tick(self):
        """One agent step: the clock moves on, and holds that ran out are released UNCHECKED."""
        self.clock += STEP_SECONDS
        for record in [r for r in self.held if r["release_at"] <= self.clock]:
            self.deliver(record, "hold ran out")

    def held_record(self, action):
        return next((r for r in self.held if r["action"] is action), None)

    def undo_one(self, action):
        """Undo one action: erase it if it is still held (it never left), else its normal undo,
        marked late if the email was already read."""
        record = self.held_record(action)
        if record:
            action.erase()
            self.held.remove(record)
            record.update(outcome="cancelled while held", delivered_at=None)
            return
        record = next((r for r in self.holds if r["action"] is action), None)
        if record and "read_at" in record:
            action.late = record["late"] = self.clock >= record["read_at"]
        action.undo()

    def hold_report(self):
        """The hold records as plain data, to save with a run."""
        return [{"action": r["action"].describe(), "type": type(r["action"]).__name__,
                 **{k: v for k, v in r.items() if k != "action"}} for r in self.holds]

    def commit(self):
        for record in list(self.held):          # the verifier said OK: release what is held
            self.deliver(record, "released after the check")
        self.journal({"event": "commit"})
        self.log(f"[UNDO] Commit: kept {len(self.completed)} action(s).")
        self.completed.clear()
        self._seqs.clear()

    def rollback(self):
        t0 = time.perf_counter()
        self.log("[UNDO] Rolling back...")
        self.journal({"event": "rollback_begin"})
        undone, stuck = 0, []
        for seq, action in reversed(list(zip(self._seqs, self.completed))):
            if action.kind == IRREVERSIBLE and not self.held_record(action):
                stuck.append(action.describe())   # a human approved it and it left; can't undo
                continue
            self.journal({"event": "undo_begin", "seq": seq})
            try:
                self.undo_one(action)
            except Exception as e:
                # One failed undo must not stop the others from running.
                stuck.append(f"{action.describe()} (undo failed: {e})")
                continue
            self.journal({"event": "undo_done", "seq": seq})
            undone += 1
            self.log(f"       undid: {action.describe()}")
        self.journal({"event": "rollback_done"})
        self.completed.clear()
        self._seqs.clear()
        self.overhead_seconds += time.perf_counter() - t0
        if stuck:
            self.log(f"[UNDO] Could NOT undo: {stuck}")
        return undone, stuck

    def rollback_some(self, chosen):
        """Step 13 (selective undo): undo ONLY the actions at these positions in self.completed
        (newest first), then commit, so every other action is kept. If the program crashes
        before the commit, recover() still undoes everything: the safe direction."""
        t0 = time.perf_counter()
        self.log(f"[UNDO] Selective rollback of {len(chosen)} of {len(self.completed)} action(s)...")
        self.journal({"event": "rollback_begin"})
        undone, stuck = 0, []
        for i in sorted(chosen, reverse=True):
            seq, action = self._seqs[i], self.completed[i]
            if action.kind == IRREVERSIBLE and not self.held_record(action):
                stuck.append(action.describe())
                continue
            self.journal({"event": "undo_begin", "seq": seq})
            try:
                self.undo_one(action)
            except Exception as e:
                stuck.append(f"{action.describe()} (undo failed: {e})")
                continue
            self.journal({"event": "undo_done", "seq": seq})
            undone += 1
            self.log(f"       undid: {action.describe()}")
        self.journal({"event": "rollback_done"})
        self.overhead_seconds += time.perf_counter() - t0
        self.commit()                                       # keep everything else
        if stuck:
            self.log(f"[UNDO] Could NOT undo: {stuck}")
        return undone, stuck


def recover(sandbox, verbose=False):
    """On restart, finish whatever a crash interrupted, using the journal in `sandbox`.

    If the task committed, keep everything. Otherwise undo every reversible action that ran and
    is not already undone, in reverse order (skipping irreversible ones, like rollback does).
    Safe to run more than once: each undo is idempotent, and a second run sees everything already
    undone. Returns (undone_count, skipped_list)."""
    records = read_journal(sandbox.journal_path)
    if any(r.get("event") == "commit" for r in records):
        return 0, []                        # the task was committed: its work stays

    actions = {}                            # seq -> the journaled action record (it ran, or tried)
    already_undone = set()                  # seq values whose undo already finished
    for r in records:
        if r.get("event") == "do_begin":
            actions[r["seq"]] = r["action"]
        elif r.get("event") == "undo_done":
            already_undone.add(r["seq"])

    undone, skipped = 0, []
    for seq in sorted(actions, reverse=True):       # reverse order, like rollback
        if seq in already_undone:
            continue
        record = actions[seq]
        if record["kind"] == IRREVERSIBLE:
            skipped.append(record["type"])
            continue
        action = from_journal(sandbox, record)
        append_journal(sandbox.journal_path, {"event": "undo_begin", "seq": seq})
        action.undo()
        append_journal(sandbox.journal_path, {"event": "undo_done", "seq": seq})
        undone += 1
        if verbose:
            print(f"[recover] undid: {action.describe()}")
    append_journal(sandbox.journal_path, {"event": "rollback_done"})
    return undone, skipped
