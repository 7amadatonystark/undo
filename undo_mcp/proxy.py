"""
undo_mcp/proxy.py — Step I5 (stretch): UNDO in front of ANOTHER MCP server.

The other server's tools declare no undo, so UNDO cannot know what they do. The safe answer
is to treat every call as IRREVERSIBLE: it is HELD until the person approves it, then
forwarded; cancelled calls never reach the other server. Every call is recorded (the flight
recorder), whatever happens to it.

    python -m undo_mcp.proxy -- <command that starts the other server>
    e.g.  python -m undo_mcp.proxy -- python -m undo_mcp

Limitation: each approved call opens a new connection to the other server, so a server that
keeps state between calls (like our own fake company) starts fresh every time. A long-lived
connection would fix that; this stretch version keeps it simple.
"""

import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.server.mcpserver import MCPServer

DOWNSTREAM = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
held = []          # calls waiting for the person: {"number", "tool", "arguments"}
flight = []        # every call and what happened to it

server = MCPServer("UNDO proxy", instructions=(
    "Every tool of the other server is HELD until the person approves it, because UNDO cannot "
    "know whether it can be undone. Never call approve_held unless the person asked you to."))


async def downstream(do):
    """Open a connection to the other server, run `do(session)`, close it."""
    params = StdioServerParameters(command=DOWNSTREAM[0], args=DOWNSTREAM[1:])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await do(session)


@server.tool()
async def list_downstream_tools() -> str:
    """The other server's tools: their names, descriptions and arguments."""
    tools = await downstream(lambda s: s.list_tools())
    return json.dumps([{"name": t.name, "description": t.description,
                        "arguments": t.input_schema} for t in tools.tools], indent=1)


@server.tool()
def call_downstream(tool: str, arguments: dict | None = None) -> str:
    """Ask to call one of the other server's tools. It is HELD until the person approves it."""
    number = len(flight) + 1
    held.append({"number": number, "tool": tool, "arguments": arguments or {}})
    flight.append({"number": number, "tool": tool, "arguments": arguments or {}, "status": "held"})
    return f"HELD #{number}: {tool} needs the person's OK (its undo is unknown)."


@server.tool()
async def approve_held(number: int) -> str:
    """Forward held call #number to the other server. ONLY when the person explicitly says so."""
    call = next((h for h in held if h["number"] == number), None)
    if call is None:
        return f"error: there is no held #{number}"
    held.remove(call)
    result = await downstream(lambda s: s.call_tool(call["tool"], call["arguments"]))
    text = " ".join(getattr(c, "text", "") for c in result.content)
    flight[number - 1].update(status="approved and sent", result=text[:300])
    return text


@server.tool()
def cancel_held(number: int) -> str:
    """Cancel held call #number: the other server never sees it."""
    call = next((h for h in held if h["number"] == number), None)
    if call is None:
        return f"error: there is no held #{number}"
    held.remove(call)
    flight[number - 1]["status"] = "cancelled"
    return f"cancelled #{number}: {call['tool']}"


@server.tool()
def flight_log() -> str:
    """Every call so far and what happened to it."""
    return json.dumps(flight, indent=1)


if __name__ == "__main__":
    if not DOWNSTREAM:
        raise SystemExit("Usage: python -m undo_mcp.proxy -- <command that starts the other server>")
    server.run()
