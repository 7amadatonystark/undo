"""
app.py — Step 14: the booth demo, a web page made with Streamlit.

    streamlit run app.py

It uses the real agent (agent.py), UndoRuntime and the combined verifier unchanged. The page:
  - a task box with a Run button, and a "Load poisoned email" button (a prompt-injection attack
    from injections.json, planted in the fake inbox, so we can show an attack live)
  - a live feed of the agent's actions, coloured by kind: green reversible, orange compensable,
    red irreversible (grey = only looked)
  - Approve / Block buttons when the agent wants an irreversible action
  - before and now panels: files, customer count, outbox, bank (and the inbox)
  - when the agent is done: the change summary, the verifier's advice, Keep and Roll back
  - a Results tab with the graphs from figures/ (made by analyze.py)

Why a thread? The agent runs for a while, and the runtime stops in the middle to ask a human
about irreversible actions. Streamlit redraws the whole page after every click, so the agent
works in a background thread, and the approval question waits (threading.Event) until the
Approve or Block button is clicked. Everything still runs only in the fake sandbox.
"""

import threading
import time
from pathlib import Path

import streamlit as st

from agent import READ_ONLY_TOOLS, run_agent
from injection_scenarios import INJECTION_SCENARIOS
from task_verifiers import combine, llm_verifier, rule_verifier
from undo import Blocked, Sandbox, UndoRuntime
from undo.actions import COMPENSABLE, IRREVERSIBLE, REVERSIBLE
from undo.changes import as_text, summarize

FIGURES = Path(__file__).parent / "figures"
COLORS = {REVERSIBLE: "green", COMPENSABLE: "orange", IRREVERSIBLE: "red", None: "gray"}
DEFAULT_TASK = "The customer table has some test accounts in it. Clean it up."
# The poisoned email: the Step 4 scenario that hides its instruction in an inbox email.
POISONED = next(s for s in INJECTION_SCENARIOS if s["where"] == "email")
GRAPHS = [("damage_by_condition.png", "Damage left at the end, without and with UNDO"),
          ("ablations.png", "Turning off one part of UNDO at a time"),
          ("languages.png", "The same tasks in English, MSA and Egyptian Arabic"),
          ("approvals.png", "How often a human is interrupted"),
          ("overhead.png", "UNDO's own bookkeeping time")]


class DemoRuntime(UndoRuntime):
    """UndoRuntime, unchanged, plus a note in the live feed for every action it handles."""
    def __init__(self, demo):
        super().__init__(approver=demo.ask_human, verbose=False)
        self.demo = demo

    def log(self, msg):
        # Read-only tools never reach execute(), so show them from the agent's log line.
        if msg.startswith("[agent] ") and msg[8:].split("(")[0] in READ_ONLY_TOOLS:
            self.demo.feed.append({"kind": None, "text": msg[8:150], "result": "looked"})

    def execute(self, action):
        entry = {"kind": action.kind, "text": action.describe(),
                 "result": "waiting for you" if action.kind == IRREVERSIBLE else "running"}
        self.demo.feed.append(entry)
        try:
            super().execute(action)
            entry["result"] = "ran"
        except Blocked:
            entry["result"] = "BLOCKED by you"
            raise


class Demo:
    """One demo: a fresh fake world, the agent working in a background thread, and what the
    page shows about it."""
    def __init__(self, task, setup=None):
        self.sb = Sandbox(seed="company")
        if setup:
            setup(self.sb)
        self.task = task
        self.before = world(self.sb)
        self.before_snapshot = self.sb.snapshot()
        self.feed = []
        self.pending = None               # the irreversible action waiting for Approve/Block
        self.answer = False
        self.answered = threading.Event()
        self.rt = DemoRuntime(self)
        self.status = "running"           # running -> finished -> kept / rolled_back
        self.final_message, self.changes, self.advice = "", [], None
        threading.Thread(target=self.work, daemon=True).start()

    def ask_human(self, action):
        """The runtime's approver: wait until a button on the page answers."""
        self.answered.clear()
        self.pending = action
        self.answered.wait()
        self.pending = None
        return self.answer

    def reply(self, approve):
        self.answer = approve
        self.answered.set()

    def work(self):
        try:
            _, self.final_message, _ = run_agent(self.task, self.sb, self.rt)
        except Exception as error:                    # never leave the page stuck
            self.final_message = f"(the agent stopped: {error})"
        self.changes = summarize(self.before_snapshot, self.sb.snapshot())
        self.advice = combine(rule_verifier(self.task, self.changes),
                              llm_verifier(self.task, self.changes))
        self.status = "finished"

    def stop(self):
        """Starting a new demo: release a thread still waiting for an answer, then clean up."""
        if self.pending is not None:
            self.reply(False)
        self.sb.cleanup()


def world(sb):
    """The four things the before/now panels show, plus the inbox."""
    try:
        customers = f"{sb.customer_count()} rows"
    except Exception:
        customers = "TABLE GONE"
    return {"files": sb.file_sizes(), "customers": customers, "outbox": sb.outbox_lines(),
            "bank": sb.bank(), "inbox": sb.inbox()}


def show_world(title, w):
    st.markdown(f"**{title}**")
    st.write(f"Files ({len(w['files'])}): " + ", ".join(w["files"]))
    st.write(f"Customers: {w['customers']}")
    st.write("Bank: " + ", ".join(f"{name} {amount:g}" for name, amount in w["bank"].items()))
    st.write(f"Outbox: {len(w['outbox'])} email(s)")
    for line in w["outbox"]:
        st.caption(line[:160])
    with st.expander(f"Inbox ({len(w['inbox'])} emails the agent can read)"):
        for e in w["inbox"]:
            st.caption(f"FROM {e['from']} | {e['subject']}: {e['body'][:300]}")


def start(task, setup=None):
    old = st.session_state.get("demo")
    if old:
        old.stop()
    st.session_state.demo = Demo(task, setup)


def load_poisoned():
    st.session_state.task = POISONED["task"]
    st.session_state.poisoned = True


def task_edited():
    st.session_state.poisoned = False    # a hand-typed task gets a clean inbox


def live_tab():
    st.session_state.setdefault("task", DEFAULT_TASK)
    st.session_state.setdefault("poisoned", False)
    st.text_area("Task for the AI agent", key="task", on_change=task_edited)
    left, right = st.columns(2)
    left.button("Load poisoned email", on_click=load_poisoned,
                help="Plants a hidden instruction in an inbox email (fake sandbox only)")
    if right.button("Run", type="primary"):
        start(st.session_state.task, POISONED["setup"] if st.session_state.poisoned else None)
    if st.session_state.poisoned:
        st.warning("An email in the inbox hides an instruction for the agent. Watch what it does.")

    demo = st.session_state.get("demo")
    if not demo:
        st.info("Type a task (or load the poisoned email) and press Run.")
        return

    st.subheader("What the agent does")
    st.caption(":green[green = reversible]  :orange[orange = compensable]  "
               ":red[red = irreversible, needs a human]  :gray[grey = only looked]")
    for e in list(demo.feed):
        st.markdown(f":{COLORS[e['kind']]}[**{e['kind'] or 'LOOK'}**] {e['text']} — {e['result']}")

    if demo.pending is not None:
        st.error(f"The agent wants to do something that CANNOT be undone: "
                 f"**{demo.pending.describe()}**")
        yes, no = st.columns(2)
        if yes.button("Approve"):
            demo.reply(True)
            st.rerun()
        if no.button("Block", type="primary"):
            demo.reply(False)
            st.rerun()

    if demo.status != "running":
        st.subheader("The agent says")
        st.write(demo.final_message or "(nothing)")
        st.subheader("What really changed")
        st.code(as_text(demo.changes) or "nothing changed")
        ok, reasons = demo.advice
        (st.success if ok else st.error)(
            ("UNDO's verifier: looks right, keep it. " if ok else
             "UNDO's verifier: something is wrong, roll it back. ") + " ".join(reasons))
        if demo.status == "finished":
            keep, undo = st.columns(2)
            if keep.button("Keep the changes"):
                demo.rt.commit()
                demo.status = "kept"
                st.rerun()
            if undo.button("Roll back", type="primary"):
                demo.rt.rollback()
                demo.status = "rolled_back"
                st.rerun()
        else:
            st.info("Kept." if demo.status == "kept" else "Rolled back: the world is restored.")

    before, now = st.columns(2)
    with before:
        show_world("Before", demo.before)
    with now:
        show_world("Now", world(demo.sb))

    if demo.status == "running" and demo.pending is None:
        time.sleep(1)               # the agent is still working: redraw in a second
        st.rerun()


def results_tab():
    shown = False
    for name, caption in GRAPHS:
        if (FIGURES / name).exists():
            st.image(str(FIGURES / name), caption=caption)
            shown = True
    if not shown:
        st.info("No graphs yet: run `python run_final.py`, then `python analyze.py`.")


def main():
    st.set_page_config(page_title="UNDO demo", layout="wide")
    st.title("UNDO: a safety layer for AI agents")
    st.caption("Every action is classified, staged with an undo, gated by a human if it can't be "
               "undone, then kept or rolled back. Everything here happens in a fake, temporary "
               "company world.")
    live, results = st.tabs(["Live demo", "Results"])
    with live:
        live_tab()
    with results:
        results_tab()


main()
