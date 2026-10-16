"""
runtime.py — the heart of UNDO.

For every action the agent attempts:
  1. CLASSIFY  - look at action.kind
  2. GATE      - irreversible actions need a human "yes" first
  3. STAGE     - prepare() saves undo info, then do() runs it
  4. COMMIT or ROLLBACK at the end:
       - commit()   keeps everything
       - rollback() undoes every completed action in REVERSE order

It also records numbers for the benchmark: how many approvals were asked,
and how much time UNDO added on top of the actions themselves.
"""

import time
from .actions import IRREVERSIBLE


class Blocked(Exception):
    """Raised when a human (or policy) refuses an irreversible action."""


class UndoRuntime:
    def __init__(self, approver, verbose=True, enabled=True):
        # approver(action) -> True/False. In the demo it asks you at the keyboard;
        # in the benchmark it is a function that plays the role of the human.
        # enabled=False is the "without UNDO" baseline: no human gate on irreversible
        # actions, so every action just runs. (rollback still works if called, but the
        # benchmark never calls it in this mode.)
        self.approver = approver
        self.verbose = verbose
        self.enabled = enabled
        self.completed = []         # stack of actions that actually ran
        self.approvals_asked = 0
        self.overhead_seconds = 0.0  # time spent in UNDO's own bookkeeping
        self.actions_seen = 0

    def log(self, msg):
        if self.verbose:
            print(msg)

    def execute(self, action):
        self.actions_seen += 1
        t0 = time.perf_counter()

        # 1-2. classify and gate. The gate only runs when UNDO is enabled;
        # without UNDO, irreversible actions are not paused for a human.
        if action.kind == IRREVERSIBLE and self.enabled:
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

        try:
            action.do()             # the real effect (not counted as overhead)
        except Exception:
            # The action may have half-happened. Put it on the stack anyway so
            # rollback() tries to undo it, then let the caller see the crash.
            self.completed.append(action)
            self.log(f"[UNDO] Agent wants: {action.describe():40s} -> CRASHED")
            raise

        self.completed.append(action)
        if action.kind != IRREVERSIBLE:
            self.log(f"[UNDO] Agent wants: {action.describe():40s} -> {action.kind}  ran")

    def commit(self):
        self.log(f"[UNDO] Commit: kept {len(self.completed)} action(s).")
        self.completed.clear()

    def rollback(self):
        t0 = time.perf_counter()
        self.log("[UNDO] Rolling back...")
        undone, stuck = 0, []
        for action in reversed(self.completed):
            if action.kind == IRREVERSIBLE:
                stuck.append(action.describe())   # a human approved it; can't undo
                continue
            try:
                action.undo()
            except Exception as e:
                # One failed undo must not stop the others from running.
                stuck.append(f"{action.describe()} (undo failed: {e})")
                continue
            undone += 1
            self.log(f"       undid: {action.describe()}")
        self.completed.clear()
        self.overhead_seconds += time.perf_counter() - t0
        if stuck:
            self.log(f"[UNDO] Could NOT undo: {stuck}")
        return undone, stuck
