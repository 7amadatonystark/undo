"""
raw_tools.py — Step L4: RAW tools, like the agents in the real incidents had.

Our normal tools are deliberately safe (filters instead of raw SQL, plain file names). This
second, harder mode gives the agent two powerful tools instead:

  run_sql(sql)       - any SQL on the sandbox's customer database
  run_shell(command) - a TINY shell written in Python. It NEVER runs a real command on this
                       computer. It knows only: ls, cat, cp, mv, rm, echo > file (or >>), and
                       mail / pay, which write to the sandbox's outbox and bank. Commands can be
                       chained with ; or &&. Anything else is refused.

Both act only inside the sandbox folder: file names must be plain names (no folders), and SQL
cannot ATTACH another file (which also blocks VACUUM INTO), load an extension, or use pragmas
other than looking at a table.
"""

import fnmatch
import os
import shlex
import shutil
import sqlite3
import uuid

from .actions import REVERSIBLE, Action, SendEmail, TransferMoney, read_outbox_line
from .changes import summarize
from .sandbox import COMPANY_ACCOUNTS, is_plain_file_name


# ---------- run_sql ----------

SAFE_PRAGMAS = {"table_info", "table_list"}


def authorizer(action, arg1, arg2, db_name, source):
    """SQLite asks this before every step of a statement. Deny anything that reaches outside
    the sandbox's database file."""
    if action in (sqlite3.SQLITE_ATTACH, sqlite3.SQLITE_DETACH):
        return sqlite3.SQLITE_DENY                    # ATTACH, DETACH and VACUUM INTO
    if action == sqlite3.SQLITE_FUNCTION and arg2 == "load_extension":
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_PRAGMA and arg1 not in SAFE_PRAGMAS:
        return sqlite3.SQLITE_DENY
    return sqlite3.SQLITE_OK


def run_sql(sandbox, sql):
    """Run SQL on the sandbox database. A single SELECT returns its rows; anything else runs as
    a script and reports how many rows it changed."""
    con = sandbox.db()
    con.set_authorizer(authorizer)
    try:
        if (sql.strip().lower().startswith(("select", "pragma"))
                and sql.strip().rstrip(";").count(";") == 0):
            cursor = con.execute(sql)
            names = [d[0] for d in cursor.description or []]
            rows = cursor.fetchall()
            return "\n".join([" | ".join(names)] + [" | ".join(str(x) for x in row) for row in rows])
        before = con.total_changes
        con.executescript(sql)
        con.commit()
        return f"done: {con.total_changes - before} row(s) changed"
    except sqlite3.Error as error:
        return f"error: {error}"
    finally:
        con.close()


# ---------- run_shell ----------

def split_commands(line):
    """'a; b && c' -> ['a', 'b', 'c'], without splitting inside quotes."""
    parts, current, quote, i = [], "", None, 0
    while i < len(line):
        ch = line[i]
        if quote:
            quote = None if ch == quote else quote
        elif ch in "'\"":
            quote = ch
        elif ch == ";" or line.startswith("&&", i):
            parts.append(current)
            current = ""
            i += 2 if ch == "&" else 1
            continue
        current += ch
        i += 1
    return [p.strip() for p in parts + [current] if p.strip()]


def file_names(sandbox, patterns):
    """Plain file names, with * patterns matched against the files that exist."""
    names = []
    for pattern in patterns:
        if not is_plain_file_name(pattern.replace("*", "x")):
            raise ValueError(f"only plain file names in this folder are allowed: {pattern!r}")
        if "*" in pattern:
            names += sorted(fnmatch.filter(os.listdir(sandbox.files_dir), pattern))
        else:
            names.append(pattern)
    return names


def shell_one(sandbox, words):
    """Run one command (already split into words). Returns its output."""
    name, args = words[0], words[1:]
    if ">" in args or ">>" in args:                       # echo text > file
        mode = "a" if ">>" in args else "w"
        i = args.index(">>" if mode == "a" else ">")
        if name != "echo" or len(args) != i + 2:
            raise ValueError("only 'echo TEXT > FILE' (or >>) can write a file")
        (target,) = file_names(sandbox, [args[i + 1]])
        with open(sandbox.path(target), mode, encoding="utf-8") as f:
            f.write(" ".join(args[:i]) + "\n")
        return ""
    if name == "ls":
        return "\n".join(f"{n} ({s} chars)" for n, s in sandbox.file_sizes().items())
    if name == "cat":
        return "\n".join(sandbox.read_file(n) or f"cat: {n}: no such file" for n in file_names(sandbox, args))
    if name == "echo":
        return " ".join(args)
    if name in ("cp", "mv"):
        source, target = file_names(sandbox, args)
        (shutil.copyfile if name == "cp" else shutil.move)(sandbox.path(source), sandbox.path(target))
        return ""
    if name == "rm":
        for n in file_names(sandbox, [a for a in args if not a.startswith("-")]):
            os.remove(sandbox.path(n))
        return ""
    if name == "mail":                                     # mail TO SUBJECT [BODY]
        to, subject, body = args[0], args[1], " ".join(args[2:])
        SendEmail(sandbox, to, subject, body).do()
        return f"mail sent to {to}"
    if name == "pay":                                      # pay AMOUNT ACCOUNT
        amount, account = float(args[0]), args[1]
        if amount <= 0:
            raise ValueError("the amount must be positive")
        TransferMoney(sandbox, amount, account).do()
        return f"paid {amount:g} to {account}"
    raise ValueError(f"unknown command {name!r} (this shell only knows: ls, cat, cp, mv, rm, "
                     "echo, mail, pay)")


def run_shell(sandbox, line):
    """Run a command line. Each command's output is joined; the first error stops the rest."""
    out = []
    for command in split_commands(line):
        try:
            out.append(shell_one(sandbox, shlex.split(command)))
        except (ValueError, IndexError, OSError) as error:
            out.append(f"error: {error}")
            break
    return "\n".join(o for o in out if o) or "done"


# ---------- a raw command as an Action UNDO can stage and undo ----------

class RawCommand(Action):
    """One run_sql or run_shell call. UNDO cannot know from its name what it will do, so its
    kind comes from a DRY RUN on a copy of the sandbox (undo/classifier.py). To undo it, a full
    copy of the local world (files and database) is saved first and put back; an email it sent
    gets a correction (compensable); money moved between the company's own accounts is moved
    back. Money that left the company cannot be undone (it is irreversible, so it was gated)."""
    kind = REVERSIBLE           # replaced for every call by the dry-run classification

    def __init__(self, sandbox, tool, text):
        super().__init__(sandbox)
        self.tool, self.text = tool, text
        self.output = ""
        self.saved = None
        self.outbox_count = 0
        self.bank_before = None

    def describe(self):
        return f"{self.tool}: {self.text[:80]}"

    def run_on(self, sandbox):
        """Run the command on any sandbox (the real one, or a copy for the dry run)."""
        return (run_sql if self.tool == "run_sql" else run_shell)(sandbox, self.text)

    def prepare(self):
        self.saved = os.path.join(self.sb.trash_dir, f"raw_{uuid.uuid4().hex}")
        shutil.copytree(self.sb.files_dir, os.path.join(self.saved, "files"))
        shutil.copyfile(self.sb.db_path, os.path.join(self.saved, "company.db"))
        self.outbox_count = len(self.sb.outbox_lines())
        self.bank_before = self.sb.bank()
        self.world_before = self.sb.snapshot()      # to compare the dry run with what happened

    def do(self):
        self.output = self.run_on(self.sb)

    def real_changes(self):
        """What the command REALLY changed (after do()), as a change summary."""
        return summarize(self.world_before, self.sb.snapshot())

    def undo(self):
        shutil.rmtree(self.sb.files_dir)
        shutil.copytree(os.path.join(self.saved, "files"), self.sb.files_dir)
        shutil.copyfile(os.path.join(self.saved, "company.db"), self.sb.db_path)
        for line in self.sb.outbox_lines()[self.outbox_count:]:
            if "CORRECTION" not in line:
                to, subject, _ = read_outbox_line(line)
                with open(self.sb.outbox_path, "a", encoding="utf-8") as f:
                    f.write(f"TO:{to} | CORRECTION: please ignore '{subject}'\n")
        bank = self.sb.bank()
        if bank != self.bank_before:
            moved = {a for a in set(bank) | set(self.bank_before)
                     if bank.get(a, 0) != self.bank_before.get(a, 0)}
            if not moved <= COMPANY_ACCOUNTS:
                raise RuntimeError("money that left the company cannot be undone")
            self.sb.save_bank(self.bank_before)
