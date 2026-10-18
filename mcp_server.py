"""
mcp_server.py — Step 12: UNDO as an MCP server, so any MCP-compatible AI app can use it.

MCP (Model Context Protocol) is a standard way for AI apps to call tools. This server offers
the same tools as our agent (agent.py), working on ONE fake, temporary sandbox, and every
change goes through UndoRuntime, exactly like in the benchmark. Two extra tools let the AI app
(or the person using it) see and control UNDO:
    undo_status - the staged actions that could still be rolled back
    rollback    - undo all of them (newest first)
    commit      - keep them (they can no longer be rolled back)

Irreversible actions (sending money, dropping the table, emailing an outside address) need a
HUMAN's OK. Over MCP, the AI app is the one calling, so it must never approve its own
actions: here they are always BLOCKED. (A future version could ask the person through the
app; see MCP.md.)

Only the fake sandbox is touched, in a temporary folder that is deleted when the server stops.

    python mcp_server.py             # run it over stdio (how AI apps start MCP servers)
See MCP.md for connecting an AI app to it.
"""

import atexit
import json

from mcp.server.mcpserver import MCPServer

from agent import run_tool
from undo import Blocked, Sandbox, UndoRuntime
from undo.actions import IRREVERSIBLE


def never_approve(action):
    """The approver for irreversible actions: the caller is an AI, so the answer is always no."""
    return False


sandbox = Sandbox(seed="company")
runtime = UndoRuntime(approver=never_approve, verbose=False)
atexit.register(sandbox.cleanup)            # the fake world disappears with the server

server = MCPServer("UNDO", instructions=(
    "Tools for a small FAKE company (files, customer database, email, bank). Every change is "
    "staged by UNDO: use undo_status to see it and rollback to undo it. Irreversible actions "
    "are blocked unless a human approves them outside this app."))


def use(tool, args=None):
    """Run one of the agent's tools through UndoRuntime. Problems become a message, never a
    crash, like agent.call_tool. Arguments left empty (None) are not passed on."""
    args = {k: v for k, v in (args or {}).items() if v is not None}
    try:
        return run_tool(tool, args, sandbox, runtime)
    except Blocked:
        return "BLOCKED: this action cannot be undone, so it needs a human's OK. It was NOT run."
    except Exception as error:
        return f"error: {error}"


# ---------- the sandbox tools (the same as the agent's) ----------

@server.tool()
def list_files() -> str:
    """List the company's files with their sizes."""
    return use("list_files")


@server.tool()
def read_file(name: str) -> str:
    """Read one file."""
    return use("read_file", {"name": name})


@server.tool()
def write_file(name: str, content: str) -> str:
    """Create or overwrite a file (reversible: UNDO keeps the old version)."""
    return use("write_file", {"name": name, "content": content})


@server.tool()
def delete_file(name: str) -> str:
    """Delete a file (reversible: UNDO keeps a hidden copy)."""
    return use("delete_file", {"name": name})


@server.tool()
def query_customers(column: str | None = None, op: str | None = None,
                    value: str | None = None) -> str:
    """List customers, optionally filtered, e.g. column='name', op='=', value='Mona Khaled'."""
    return use("query_customers", {"column": column, "op": op, "value": value})


@server.tool()
def delete_rows(column: str | None = None, op: str | None = None,
                value: str | None = None) -> str:
    """Delete customers matching a filter. No filter deletes EVERY row (reversible)."""
    return use("delete_rows", {"column": column, "op": op, "value": value})


@server.tool()
def send_email(to: str, subject: str, body: str | None = None) -> str:
    """Send an email (fake outbox). To known people it can be corrected; to an outside
    address it is irreversible and needs a human."""
    return use("send_email", {"to": to, "subject": subject, "body": body})


@server.tool()
def transfer_money(amount: float, to: str) -> str:
    """Transfer money from the company account (irreversible: needs a human)."""
    return use("transfer_money", {"amount": amount, "to": to})


# ---------- UNDO's own tools ----------

@server.tool()
def undo_status() -> str:
    """The staged actions that could still be rolled back, oldest first."""
    if not runtime.completed:
        return "Nothing staged: there is nothing to roll back."
    lines = [f"{i + 1}. {a.describe()} ({a.kind}"
             + (", cannot be undone)" if a.kind == IRREVERSIBLE else ")")
             for i, a in enumerate(runtime.completed)]
    return f"{len(lines)} staged action(s):\n" + "\n".join(lines)


@server.tool()
def rollback() -> str:
    """Undo every staged action, newest first."""
    undone, stuck = runtime.rollback()
    return json.dumps({"undone": undone, "could_not_undo": stuck})


@server.tool()
def commit() -> str:
    """Keep every staged action. After this they can no longer be rolled back."""
    kept = len(runtime.completed)
    runtime.commit()
    return f"Kept {kept} action(s)."


if __name__ == "__main__":
    server.run()                # stdio: the AI app starts this program and talks over stdin/stdout
