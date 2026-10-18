"""
runtime.py — the heart of UNDO.

For every action the agent attempts:
  1. CLASSIFY  - look at action.kind
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

It also records numbers for the benchmark: how many approvals were asked, and how much time
UNDO added on top of the actions themselves (now including the journal's fsyncs).
"""

import json
import os
import time

from .actions import IRREVERSIBLE, from_journal, to_journal


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
    def __init__(self, approver, verbose=True, enabled=True, gate=True):
        # approver(action) -> True/False. In the demo it asks you at the keyboard;
        # in the benchmark it is a function that plays the role of the human.
        # enabled=False is the "without UNDO" baseline: no human gate on irreversible
        # actions, so every action just runs. (rollback still works if called, but the
        # benchmark never calls it in this mode.)
        # gate=False is a Step 6 ablation: UNDO still stages and can roll back, but irreversible
        # actions are NOT paused for a human. It shows how much the gate alone contributes.
        self.approver = approver
        self.gate = gate
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

        # 1-2. classify and gate. The gate only runs when UNDO is enabled;
        # without UNDO, irreversible actions are not paused for a human.
        if action.kind == IRREVERSIBLE and self.enabled and self.gate:
            self.approvals_asked += 1
            self.log(f"[UNDO] Agent wants: {action.describe():40s} -> IRREVERSIBLE  paused")
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
        if action.kind != IRREVERSIBLE:
            self.log(f"[UNDO] Agent wants: {action.describe():40s} -> {action.kind}  ran")

    def commit(self):
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
            if action.kind == IRREVERSIBLE:
                stuck.append(action.describe())   # a human approved it; can't undo
                continue
            self.journal({"event": "undo_begin", "seq": seq})
            try:
                action.undo()
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
            if action.kind == IRREVERSIBLE:
                stuck.append(action.describe())
                continue
            self.journal({"event": "undo_begin", "seq": seq})
            try:
                action.undo()
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
