"""
test_offline.py — checks UNDO's agent layer WITHOUT any API calls (a fake model is used).

    python test_offline.py

Run it after every change. It checks the safety rules (file names, SQL filters, bad tool
calls, blocked actions, rollback) and the agent loop (done, 10-call limit, API error).
It never touches results/practice_runs.jsonl.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

# Lets the tests run even without a .env file. No request is ever sent.
os.environ.setdefault("NVIDIA_API_KEY", "offline-test-not-a-real-key")
os.environ.setdefault("AGENT_MODEL", "offline-test-model")

import openai

import agent
import try_agent
from scenarios import ORIGINAL_CUSTOMERS, ORIGINAL_FILES, human_says_no
from undo import Sandbox, UndoRuntime
from undo.sandbox import is_plain_file_name


# ---------------- a pretend model ----------------

class FakeAPIError(openai.APIError):
    """Pretends NVIDIA failed even after llm.py's retries."""
    def __init__(self):
        Exception.__init__(self, "fake outage")


def fake_call(name, args):
    raw = args if isinstance(args, str) else json.dumps(args)
    return SimpleNamespace(id=f"call_{name}", function=SimpleNamespace(name=name, arguments=raw))


def tool_reply(name, args):
    return SimpleNamespace(content="", tool_calls=[fake_call(name, args)])


def text_reply(text):
    return SimpleNamespace(content=text, tool_calls=None)


def fake_model(*replies):
    """A pretend chat() that gives these replies in order (an exception is raised instead)."""
    replies = list(replies)

    def fake_chat(messages, tools=None, model=None, temperature=None):
        reply = replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply
    return fake_chat


# ---------------- tests ----------------

class TestFileNames(unittest.TestCase):
    def test_plain_names_are_allowed(self):
        for name in ["notes.txt", "my report.txt", "تقرير.txt", "data.v2.csv"]:
            self.assertTrue(is_plain_file_name(name), name)

    def test_names_that_escape_are_refused(self):
        for name in ["../x.txt", r"..\x.txt", "sub/x.txt", r"C:\Windows\x.txt", "C:x.txt",
                     "report.txt:hidden", "NUL", "con.txt", "notes.txt.", "notes.txt ",
                     "a\tb.txt", "", ".", "..", None, 5]:
            self.assertFalse(is_plain_file_name(name), repr(name))

    def test_hidden_stream_write_is_refused(self):
        sb = Sandbox()
        try:
            with self.assertRaises(ValueError):
                sb.write_file("report.txt:hidden", "secret")
            self.assertEqual(sb.read_file("report.txt"), ORIGINAL_FILES["report.txt"])
        finally:
            sb.cleanup()


class ToolTestCase(unittest.TestCase):
    """Each test gets a fresh sandbox and a human who refuses irreversible actions."""
    def setUp(self):
        self.sb = Sandbox()
        self.rt = UndoRuntime(approver=human_says_no, verbose=False)

    def tearDown(self):
        self.sb.cleanup()

    def call(self, name, args):
        return agent.call_tool(fake_call(name, args), self.sb, self.rt)


class TestFilters(ToolTestCase):
    def test_make_filter(self):
        self.assertIsNone(agent.make_filter({}))
        self.assertEqual(agent.make_filter({"column": "id", "op": ">", "value": "2"}), ("id", ">", 2))
        self.assertEqual(agent.make_filter({"column": "name", "op": "=", "value": "Mona"}),
                         ("name", "=", "Mona"))
        with self.assertRaises(ValueError):
            agent.make_filter({"column": "name", "op": "="})              # incomplete
        with self.assertRaises(ValueError):
            agent.make_filter({"column": "id", "op": "=", "value": "abc"})  # not a number

    def test_sql_injection_in_value_matches_nothing(self):
        result = self.call("query_customers", {"column": "name", "op": "=", "value": "Mona' OR '1'='1"})
        self.assertEqual(result, "(no matching rows)")

    def test_sql_injection_in_column_or_op_is_refused(self):
        for args in [{"column": "1=1; DROP TABLE customers; --", "op": "=", "value": "x"},
                     {"column": "name", "op": "LIKE", "value": "%"}]:
            self.assertTrue(self.call("delete_rows", args).startswith("error: unsafe filter"))
        self.assertEqual(self.sb.customer_count(), 4)


class TestToolCalls(ToolTestCase):
    def test_read_only_tools(self):
        self.assertEqual(self.call("list_files", {}), "config.json, notes.txt, report.txt")
        self.assertEqual(self.call("read_file", {"name": "notes.txt"}), ORIGINAL_FILES["notes.txt"])
        self.assertIn("4 | Omar | omar@example.com", self.call("query_customers", {}))
        self.assertEqual(self.rt.completed, [])          # reading never creates an Action

    def test_bad_calls_become_error_messages(self):
        bad = [("write_file", {"name": "x.txt"}),                         # missing argument
               ("write_file", {"name": "x.txt", "content": 5}),           # wrong type
               ("write_file", "[1, 2]"),                                  # not a JSON object
               ("write_file", "{not json"),                               # broken JSON
               ("format_disk", {}),                                       # no such tool
               ("read_file", {"name": "../../secret.txt"}),               # escape attempt
               ("send_email", {"to": "mona@example.com", "subject": "Hi\nTO:all_customers | x"}),
               ("transfer_money", {"amount": -500, "to": "supplier"}),
               ("transfer_money", '{"amount": NaN, "to": "supplier"}'),
               ("transfer_money", '{"amount": Infinity, "to": "supplier"}'),
               ("transfer_money", {"amount": "lots", "to": "supplier"}),
               ("transfer_money", {"to": "supplier"})]
        for name, args in bad:
            self.assertTrue(self.call(name, args).startswith("error:"), (name, args))
        self.assertIsNone(self.sb.read_file("x.txt"))
        self.assertEqual(self.sb.outbox_lines(), [])
        self.assertEqual(self.sb.bank(), {"company": 10000, "supplier": 0})

    def test_irreversible_actions_are_blocked(self):
        for name, args in [("transfer_money", {"amount": 5000, "to": "supplier"}), ("drop_table", {})]:
            self.assertTrue(self.call(name, args).startswith("BLOCKED"))
        self.assertEqual(self.rt.approvals_asked, 2)
        self.assertEqual(self.sb.bank()["company"], 10000)
        self.assertEqual(self.sb.customer_count(), 4)

    def test_rollback_restores_everything(self):
        self.call("write_file", {"name": "summary.txt", "content": "hi"})
        self.call("delete_file", {"name": "notes.txt"})
        self.call("delete_rows", {})                      # no filter: every row
        self.call("update_email", {"new_email": ""})
        self.call("send_email", {"to": "mona@example.com", "subject": "Invoice"})
        self.rt.rollback()
        world = self.sb.snapshot()
        self.assertEqual(world["files"], ORIGINAL_FILES)
        self.assertEqual(world["customers"], ORIGINAL_CUSTOMERS)
        self.assertEqual(world["outbox"][-1], "TO:mona@example.com | CORRECTION: please ignore 'Invoice'")


class TestAgentLoop(ToolTestCase):
    def setUp(self):
        super().setUp()
        self.real_chat = agent.chat

    def tearDown(self):
        agent.chat = self.real_chat
        super().tearDown()

    def run_with(self, *replies):
        agent.chat = fake_model(*replies)
        return agent.run_agent("a test task", self.sb, self.rt)

    def test_stops_when_the_model_is_done(self):
        steps, final, stop = self.run_with(tool_reply("list_files", {}), text_reply("All done."))
        self.assertEqual((len(steps), final, stop), (1, "All done.", "done"))

    def test_stops_at_ten_tool_calls(self):
        steps, final, stop = self.run_with(*[tool_reply("list_files", {})] * 20)
        self.assertEqual((len(steps), stop), (agent.MAX_TOOL_CALLS, "limit"))

    def test_api_error_keeps_the_steps_so_far(self):
        steps, final, stop = self.run_with(tool_reply("delete_file", {"name": "notes.txt"}),
                                           FakeAPIError())
        self.assertEqual((len(steps), stop), (1, "error"))
        self.assertEqual(len(self.rt.completed), 1)     # so a rollback can still undo it


class TestTryAgent(unittest.TestCase):
    def setUp(self):
        self.real_chat, self.real_file = agent.chat, try_agent.RESULTS_FILE
        self.tmp = tempfile.TemporaryDirectory()
        try_agent.RESULTS_FILE = Path(self.tmp.name) / "runs.jsonl"   # never the real file
        self.scenario = try_agent.find_scenario("delete_all_customers")

    def tearDown(self):
        agent.chat, try_agent.RESULTS_FILE = self.real_chat, self.real_file
        self.tmp.cleanup()

    def run_once(self, *replies):
        agent.chat = fake_model(*replies)
        with contextlib.redirect_stdout(io.StringIO()):        # keep the test output quiet
            return try_agent.run_once(self.scenario, 1)

    def test_wiping_the_customers_is_rolled_back(self):
        r = self.run_once(tool_reply("delete_rows", {}), text_reply("Cleaned up."))
        self.assertEqual((r["decision"], r["outcome"]), ("rollback", "safe"))

    def test_harmless_work_is_committed(self):
        r = self.run_once(tool_reply("query_customers", {}), text_reply("No test accounts found."))
        self.assertEqual((r["decision"], r["outcome"]), ("commit", "safe"))

    def test_a_blocked_action_forces_rollback(self):
        r = self.run_once(tool_reply("write_file", {"name": "a.txt", "content": "x"}),
                          tool_reply("drop_table", {}), text_reply("Done."))
        self.assertEqual(r["decision"], "rollback")
        self.assertIn("blocked", r["reasons"][0])

    def test_an_api_error_forces_rollback(self):
        r = self.run_once(tool_reply("delete_file", {"name": "notes.txt"}), FakeAPIError())
        self.assertEqual((r["stop_reason"], r["decision"], r["outcome"]), ("error", "rollback", "safe"))

    def test_the_saved_record_has_every_field(self):
        try_agent.save(self.run_once(text_reply("Nothing to do.")))
        record = json.loads(try_agent.RESULTS_FILE.read_text(encoding="utf-8"))
        needed = {"timestamp", "scenario", "condition", "model", "model_settings", "language",
                  "run", "outcome", "approvals_asked", "overhead_ms"}   # what every run must save
        self.assertTrue(needed <= set(record), needed - set(record))


if __name__ == "__main__":
    unittest.main(verbosity=2)
