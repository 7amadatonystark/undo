"""
undo_mcp/server.py — Step I5: UNDO as an MCP server for MCP-compatible AI apps.

MCP (Model Context Protocol) is a standard way for AI apps to call tools. This server offers a
small FAKE company (files, customers, inbox, outbox, bank) in a temporary folder it creates on
startup and deletes on exit. It never touches real files, email or money, and it needs no API
key: it only uses the `undo` package.

Every call goes through UndoRuntime, with:
  - the per-call classifier (Step I1): the same tool can be reversible or irreversible
  - the hold window (Step I2): emails and payments are HELD instead of sent. Compensable ones
    are released after HOLD_SECONDS of simulated time (one call = one step); IRREVERSIBLE ones
    (an email to a stranger, money to an outside account, a leak) wait for approve_held
  - the flight recorder (Step 5): every call, with its kind and result, is logged

Control tools: undo_status, rollback, approve_held, cancel_held, flight_log.

approve_held is for the PERSON using the app: an AI must never approve its own irreversible
action. The tool's description says so, and most MCP apps show each tool call to the person
before running it. (That is a trust boundary this server cannot enforce by itself.)

    python -m undo_mcp          # from the project folder; see QUICKSTART.md
"""

import atexit
import json
import math

from mcp.server.mcpserver import MCPServer

from undo import Blocked, Sandbox, UndoRuntime
from undo.actions import (IRREVERSIBLE, DeleteFile, DeleteRows, SendEmail, TransferMoney,
                          WriteFile, where_clause)
from undo.settings import HOLD_SECONDS

NEEDS_A_PERSON = math.inf          # release time of an irreversible held effect


def never_approve(action):
    return False                   # (not used: irreversible effects are held, not gated)


sandbox = Sandbox(seed="company")
atexit.register(sandbox.cleanup)    # the fake world disappears with the server
# gate=False: instead of refusing irreversible calls, the server HOLDS their effects until the
# person approves them (approve_held) or cancels them.
runtime = UndoRuntime(approver=never_approve, verbose=False, gate=False, hold_seconds=HOLD_SECONDS)
flight = []                         # every call: the flight recorder

server = MCPServer("UNDO", instructions=(
    "Tools for a small FAKE company (files, customer database, inbox, outbox, bank). Every "
    "change is staged by UNDO: undo_status shows what was done and what is held, rollback "
    "undoes it. Emails and payments are HELD; irreversible ones wait for the person to approve "
    "them. Never call approve_held unless the person explicitly asked you to."))


def record(tool, args, result, action=None):
    flight.append({"call": len(flight) + 1, "tool": tool, "args": args, "result": result[:300],
                   "kind": getattr(action, "kind", "read-only")})
    return result


def change(tool, args, make):
    """Run one change through UndoRuntime and hold irreversible effects for a person."""
    runtime.tick()                  # one call = one step on the simulated clock
    try:
        action = make()
        runtime.execute(action)
    except Blocked:
        return record(tool, args, "BLOCKED: it needs a person's OK. It was not run.")
    except Exception as error:
        return record(tool, args, f"error: {error}")
    held = runtime.held_record(action)
    if held and action.kind == IRREVERSIBLE:
        held["release_at"] = NEEDS_A_PERSON
        return record(tool, args, f"HELD until a person approves it (it cannot be undone once "
                                  f"it leaves): {action.describe()}", action)
    if held:
        return record(tool, args, f"held for {HOLD_SECONDS:g} s of simulated time, then sent: "
                                  f"{action.describe()}", action)
    return record(tool, args, f"done: {action.describe()}", action)


def look(tool, args, result):
    runtime.tick()
    return record(tool, args, result)


def filter_of(column, op, value):
    if column is None and op is None and value is None:
        return None                 # every row
    if None in (column, op, value):
        raise ValueError("a filter needs column, op and value together")
    return (column, op, value)


# ---------- the sandbox tools ----------

@server.tool()
def list_files() -> str:
    """List the company's files with their sizes."""
    return look("list_files", {}, "\n".join(f"{n} ({s} chars)" for n, s in sandbox.file_sizes().items()))


@server.tool()
def read_file(name: str) -> str:
    """Read one file (a plain file name, e.g. notes.txt)."""
    try:
        text = sandbox.read_file(name)
    except Exception as error:
        text = f"error: {error}"
    return look("read_file", {"name": name}, "(no such file)" if text is None else text)


@server.tool()
def read_inbox() -> str:
    """Read the emails in the inbox."""
    return look("read_inbox", {}, "\n\n".join(f"FROM: {e['from']}\nSUBJECT: {e['subject']}\n{e['body']}"
                                              for e in sandbox.inbox()))


@server.tool()
def query_customers(column: str | None = None, op: str | None = None, value: str | None = None) -> str:
    """List customers, optionally filtered, e.g. column='name', op='=', value='Mona Khaled'."""
    try:
        sql, params = where_clause(filter_of(column, op, value))
        con = sandbox.db()
        try:
            rows = con.execute(f"SELECT id, name, email FROM customers WHERE {sql} ORDER BY id",
                               params).fetchall()
        finally:
            con.close()
        text = "id | name | email\n" + "\n".join(f"{i} | {n} | {e}" for i, n, e in rows)
    except Exception as error:
        text = f"error: {error}"
    return look("query_customers", {"column": column, "op": op, "value": value}, text)


@server.tool()
def write_file(name: str, content: str) -> str:
    """Create or overwrite a file (reversible: UNDO keeps the old version)."""
    return change("write_file", {"name": name}, lambda: WriteFile(sandbox, name, content))


@server.tool()
def delete_file(name: str) -> str:
    """Delete a file (reversible: UNDO keeps a hidden copy)."""
    return change("delete_file", {"name": name}, lambda: DeleteFile(sandbox, name))


@server.tool()
def delete_rows(column: str | None = None, op: str | None = None, value: str | None = None) -> str:
    """Delete customers matching a filter. No filter deletes EVERY row (reversible)."""
    return change("delete_rows", {"column": column, "op": op, "value": value},
                  lambda: DeleteRows(sandbox, filter_of(column, op, value)))


@server.tool()
def send_email(to: str, subject: str, body: str = "") -> str:
    """Send an email (fake outbox). It is held first; an irreversible one waits for a person."""
    return change("send_email", {"to": to, "subject": subject},
                  lambda: SendEmail(sandbox, to, subject, body))


@server.tool()
def transfer_money(amount: float, to: str) -> str:
    """Transfer money from the company account. Held; money leaving the company waits for a person."""
    def make():
        if not math.isfinite(amount) or amount <= 0:
            raise ValueError("amount must be a number more than 0")
        return TransferMoney(sandbox, amount, to)
    return change("transfer_money", {"amount": amount, "to": to}, make)


# ---------- UNDO's own tools ----------

@server.tool()
def undo_status() -> str:
    """What was done (and can still be undone), and what is held."""
    done = [f"{i + 1}. {a.describe()} ({a.kind})" for i, a in enumerate(runtime.completed)]
    held = [f"held #{n}: {r['action'].describe()} - "
            + ("waits for a person" if r["release_at"] == NEEDS_A_PERSON
               else f"sent at t={r['release_at']:g}s (now t={runtime.clock:g}s)")
            for n, r in enumerate(runtime.held, 1)]
    return "\n".join(["Done (can be rolled back):"] + (done or ["  nothing"])
                     + ["Held (not sent yet):"] + (held or ["  nothing"]))


@server.tool()
def rollback() -> str:
    """Undo everything done so far, newest first. Held effects are cancelled; nothing leaves."""
    undone, stuck = runtime.rollback()
    return json.dumps({"undone": undone, "could_not_undo": stuck})


@server.tool()
def approve_held(number: int) -> str:
    """Send held effect #number now. ONLY when the person explicitly says so."""
    if not 1 <= number <= len(runtime.held):
        return f"error: there is no held #{number}"
    record_ = runtime.held[number - 1]
    runtime.deliver(record_, "approved by the person")
    return f"sent: {record_['action'].describe()}"


@server.tool()
def cancel_held(number: int) -> str:
    """Cancel held effect #number: it never leaves, as if it never happened."""
    if not 1 <= number <= len(runtime.held):
        return f"error: there is no held #{number}"
    action = runtime.held[number - 1]["action"]
    runtime.undo_one(action)
    i = runtime.completed.index(action)
    del runtime.completed[i], runtime._seqs[i]
    return f"cancelled: {action.describe()}"


@server.tool()
def flight_log() -> str:
    """Every call made so far (the flight recorder), with its kind and result."""
    return json.dumps(flight, indent=1)


def main():
    server.run()                    # stdio: the AI app starts this program and talks over it
