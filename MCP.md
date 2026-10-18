# UNDO as an MCP server (Step 12)

[MCP](https://modelcontextprotocol.io) (Model Context Protocol) is a standard way for AI apps to
call tools. `mcp_server.py` makes UNDO's fake company world available to any app that speaks
MCP. The app's AI gets the same tools as our benchmark agent, and **every change goes through
UndoRuntime**, so it is staged and can be rolled back.

## Tools

| Tool | What it does | Kind |
| --- | --- | --- |
| `list_files`, `read_file`, `query_customers` | look around | read-only |
| `write_file`, `delete_file`, `delete_rows` | change files / customers | reversible |
| `send_email` | fake outbox | compensable to known people, irreversible to outside addresses |
| `transfer_money` | fake bank | irreversible |
| `undo_status` | list the staged actions | UNDO |
| `rollback` | undo every staged action, newest first | UNDO |
| `commit` | keep them (no rollback after this) | UNDO |

**Irreversible actions are always blocked over MCP.** UNDO's rule is that a *human* approves
them, but over MCP the caller is the AI app itself, and an AI must never approve its own
actions. A future version could ask the person through the app (MCP "elicitation").

The world is a fake, temporary sandbox in your temp folder, deleted when the server stops. It
never touches real files, email or money.

## Run it

The server talks over stdio: the AI app starts it as a program. Use the full path to the
virtual environment's Python, so the right packages are found:

```
C:\Users\abdul\Documents\undo_project\.venv\Scripts\python.exe C:\Users\abdul\Documents\undo_project\mcp_server.py
```

(`config.py` loads `.env` from the project folder, so the NVIDIA key must be there, but the
server itself makes no AI calls.)

## Connect an AI app

Most MCP-compatible AI apps have a settings file or screen for MCP servers. Add a server named
`undo` that runs this command (the usual JSON form is shown):

```json
{
  "mcpServers": {
    "undo": {
      "command": "C:\\Users\\abdul\\Documents\\undo_project\\.venv\\Scripts\\python.exe",
      "args": ["C:\\Users\\abdul\\Documents\\undo_project\\mcp_server.py"]
    }
  }
}
```

Apps with a command line usually also have an "add MCP server" command that takes the same
program and argument.

## Try it

Ask the AI app, for example:
1. "Use the undo tools: delete report.txt, then show me the undo status." → one staged action.
2. "Now roll it back and list the files." → report.txt is back.
3. "Transfer 5000 to the supplier." → BLOCKED: it needs a human.

## How it was tested

`test_offline.py` (TestMCPServer) calls the tools inside Python. A second check started
`mcp_server.py` as a separate program and talked to it over stdio with the official MCP Python
client: it listed all 11 tools, and delete → undo_status → rollback restored the file.
Testing with a real AI app is still to do.
