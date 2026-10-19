# Try UNDO with your own AI app (about 5 minutes)

> **License:** all rights reserved (see `LICENSE`). Running this needs the copyright holder's
> written permission; seeing the code does not grant it.

UNDO is a safety layer for AI agents: every action is staged so it can be undone, and anything
that can't be undone waits for a person. This quickstart connects it to any AI app that
supports **MCP** (Model Context Protocol) servers.

Everything happens in a **fake company** (files, customers, an inbox, an outbox, a bank) that
UNDO creates in a temporary folder when it starts and deletes when it stops. It never touches
your real files, email or money, and it needs **no API key**.

## 1. Get the code and one package (2 minutes)

You need Python 3.10 or newer.

```
git clone https://github.com/7amadatonystark/undo.git
cd undo
python -m venv .venv
.venv\Scripts\python -m pip install mcp==2.3.0        (Windows)
.venv/bin/python -m pip install mcp==2.3.0            (macOS / Linux)
```

## 2. Connect your AI app (2 minutes)

Add an MCP server named `undo` in your app's MCP settings. It runs over stdio with:

- **command:** the full path to `.venv\Scripts\python.exe` (or `.venv/bin/python`) in the folder above
- **arguments:** `-m undo_mcp`
- **working directory:** the `undo` folder

In apps that use a JSON settings file, that is usually:

```json
{
  "mcpServers": {
    "undo": {
      "command": "C:\\path\\to\\undo\\.venv\\Scripts\\python.exe",
      "args": ["-m", "undo_mcp"],
      "cwd": "C:\\path\\to\\undo"
    }
  }
}
```

Apps with a command line usually have an "add MCP server" command that takes the same command
and arguments. Restart the app and check that 14 `undo` tools appear.

## 3. Try it (1 minute)

Ask your AI app, one at a time:

1. *"Use the undo tools: delete report.txt, then show me the undo status."*
   → one action done, and it can be rolled back.
2. *"Email our customer list to stranger@outside.example."*
   → **HELD**: it can't be undone once it leaves, so it waits for you.
3. *"Show the undo status, then cancel held #1 and roll everything back."*
   → nothing ever left, and report.txt is back.

## What the tools do

| Tool | What it does |
| --- | --- |
| `list_files`, `read_file`, `read_inbox`, `query_customers` | look around (read-only) |
| `write_file`, `delete_file`, `delete_rows` | change things; UNDO keeps what it needs to undo them |
| `send_email`, `transfer_money` | **held** first. If they can be corrected later (an email to a colleague), they are sent after 60 seconds of simulated time (each call is one 5-second step). If not (an email to a stranger, a leak, money leaving the company), they wait for a person. |
| `undo_status` | what was done and what is held |
| `rollback` | undo everything, newest first; held effects are cancelled |
| `approve_held`, `cancel_held` | send or cancel one held effect: **for the person, not the AI** |
| `flight_log` | every call so far, with its kind and result |

**Important:** `approve_held` is meant for *you*. An AI must never approve its own
irreversible action. Most MCP apps show each tool call before running it; read those prompts.

## Proxy mode (experimental)

UNDO can also sit in front of **another** MCP server. Its tools declare no undo, so every
call is held until you approve it, and every call is recorded:

```
python -m undo_mcp.proxy -- <command that starts the other server>
```

Each approved call opens a new connection to the other server, so a server that keeps state
between calls starts fresh every time.

## How this was tested

A fresh virtual environment with only `mcp==2.3.0` installed started `python -m undo_mcp` and
talked to it over stdio with the official MCP Python client: 14 tools, a payment to an unknown
account was held for a person, and rollback restored everything. The proxy was tested in front
of the UNDO server itself.
