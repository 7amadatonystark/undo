"""
actions.py — every kind of thing an agent can do, and how to undo it.

Each Action has:
  - kind:      REVERSIBLE, COMPENSABLE or IRREVERSIBLE
  - prepare(): runs BEFORE the action, saves whatever is needed to undo it
  - do():      actually performs the action on the sandbox
  - undo():    reverses it (or, for compensable actions, sends a fix)

This is the Saga pattern: every step that can go wrong carries its own undo.
"""

import json
import os
import shutil
import sqlite3
import uuid

from .sandbox import COMPANY_ACCOUNTS

REVERSIBLE = "REVERSIBLE"        # a real inverse exists (restore the old file)
COMPENSABLE = "COMPENSABLE"      # no true undo, but a correcting action exists
IRREVERSIBLE = "IRREVERSIBLE"    # nothing can undo it -> a human must approve first


# ---------------- safe row filters ----------------
# Database actions take a filter like ("name", "=", "Mona") instead of raw SQL text.
# The column and operator must come from these lists, and the value is passed to
# SQLite as a parameter (?), so an agent can never smuggle its own SQL in.

ALLOWED_COLUMNS = {"id", "name", "email"}
ALLOWED_OPS = {"=", "!=", "<", "<=", ">", ">="}


def where_clause(where):
    """(column, op, value) -> ("column op ?", (value,)). None means every row."""
    if where is None:
        return "1=1", ()
    column, op, value = where
    if column not in ALLOWED_COLUMNS or op not in ALLOWED_OPS:
        raise ValueError(f"unsafe filter: {where!r}")
    return f"{column} {op} ?", (value,)


def describe_where(where):
    if where is None:
        return "(every row)"
    column, op, value = where
    return f"{column} {op} {value!r}"


class Action:
    kind = REVERSIBLE

    def __init__(self, sandbox):
        self.sb = sandbox

    def describe(self):
        return self.__class__.__name__

    def prepare(self):
        """Save what we need for undo. Called just before do()."""

    def do(self):
        raise NotImplementedError

    def undo(self):
        raise NotImplementedError


# ---------------- REVERSIBLE ----------------

class WriteFile(Action):
    kind = REVERSIBLE

    def __init__(self, sandbox, name, content):
        super().__init__(sandbox)
        self.name, self.content = name, content
        self.old_content = None

    def describe(self):
        return f"write file '{self.name}'"

    def prepare(self):
        self.old_content = self.sb.read_file(self.name)  # None if the file was new

    def do(self):
        self.sb.write_file(self.name, self.content)

    def undo(self):
        if self.old_content is None:
            p = self.sb.path(self.name)
            if os.path.exists(p):                       # file didn't exist before; remove it
                os.remove(p)                            # (guarded so recovery can run twice)
        else:
            self.sb.write_file(self.name, self.old_content)


class DeleteFile(Action):
    kind = REVERSIBLE

    def __init__(self, sandbox, name):
        super().__init__(sandbox)
        self.name = name
        # Unique name, so deleting the same file twice never overwrites a backup.
        self.trash_copy = os.path.join(sandbox.trash_dir, f"{uuid.uuid4().hex}_{name}")

    def describe(self):
        return f"delete file '{self.name}'"

    def do(self):
        # Instead of truly deleting, move the file into a hidden trash folder.
        shutil.move(self.sb.path(self.name), self.trash_copy)

    def undo(self):
        if not os.path.exists(self.trash_copy):
            return                                      # do() crashed before moving it
        shutil.move(self.trash_copy, self.sb.path(self.name))


class DeleteRows(Action):
    """Delete database rows. Reversible because we copy the rows first."""
    kind = REVERSIBLE

    def __init__(self, sandbox, where=None):
        super().__init__(sandbox)
        self.where = where              # e.g. ("id", ">", 2); None = every row
        self.saved_rows = []

    def describe(self):
        return f"DELETE FROM customers WHERE {describe_where(self.where)}"

    def prepare(self):
        sql, params = where_clause(self.where)
        con = self.sb.db()
        self.saved_rows = con.execute(
            f"SELECT id, name, email FROM customers WHERE {sql}", params).fetchall()
        con.close()

    def do(self):
        sql, params = where_clause(self.where)
        con = self.sb.db()
        con.execute(f"DELETE FROM customers WHERE {sql}", params)
        con.commit()
        con.close()

    def undo(self):
        con = self.sb.db()
        # OR REPLACE: safe even if do() crashed and some rows were never deleted.
        con.executemany("INSERT OR REPLACE INTO customers (id, name, email) VALUES (?, ?, ?)",
                        self.saved_rows)
        con.commit()
        con.close()


class AddCustomer(Action):
    """Insert one customer. Undo deletes that exact row again."""
    kind = REVERSIBLE

    def __init__(self, sandbox, name, email):
        super().__init__(sandbox)
        self.name, self.email = name, email
        self.before_max_id = 0

    def describe(self):
        return f"add customer '{self.name}'"

    def prepare(self):
        # The highest id BEFORE we insert. Undo removes the matching row(s) added after this,
        # so recovery works even if a crash lost the new row's id. (Captured in prepare so it is
        # written to the journal before do() runs.)
        con = self.sb.db()
        self.before_max_id = con.execute("SELECT MAX(id) FROM customers").fetchone()[0] or 0
        con.close()

    def do(self):
        con = self.sb.db()
        con.execute("INSERT INTO customers (name, email) VALUES (?, ?)", (self.name, self.email))
        con.commit()
        con.close()

    def undo(self):
        # Delete the row(s) this action added: newer than before_max_id, with our name+email.
        # Safe to run twice (the second time matches nothing) and after a crash mid-insert.
        con = self.sb.db()
        con.execute("DELETE FROM customers WHERE id > ? AND name = ? AND email = ?",
                    (self.before_max_id, self.name, self.email))
        con.commit()
        con.close()


class SetEmail(Action):
    """Change the email of matching customers. Reversible: old emails are saved."""
    kind = REVERSIBLE

    def __init__(self, sandbox, where, new_email):
        super().__init__(sandbox)
        self.where, self.new_email = where, new_email   # where: like DeleteRows
        self.old_emails = []

    def describe(self):
        return f"UPDATE customers SET email={self.new_email!r} WHERE {describe_where(self.where)}"

    def prepare(self):
        sql, params = where_clause(self.where)
        con = self.sb.db()
        self.old_emails = con.execute(
            f"SELECT id, email FROM customers WHERE {sql}", params).fetchall()
        con.close()

    def do(self):
        sql, params = where_clause(self.where)
        con = self.sb.db()
        con.execute(f"UPDATE customers SET email = ? WHERE {sql}",
                    (self.new_email,) + params)
        con.commit()
        con.close()

    def undo(self):
        con = self.sb.db()
        con.executemany("UPDATE customers SET email = ? WHERE id = ?",
                        [(email, cid) for cid, email in self.old_emails])
        con.commit()
        con.close()


# ---------------- COMPENSABLE ----------------

COMPANY_DOMAIN = "@company.example"     # the company's own email addresses


class SendEmail(Action):
    """You can't un-send an email. You CAN send a correction, but only to people we know.

    By default an email is COMPENSABLE: a correction reaches the same people. But the kind of
    ONE email depends on who it goes to and what it carries, so undo/classifier.py decides each
    call (Step I1): an email to an unknown outside address, or one carrying a secret or other
    customers' data, is IRREVERSIBLE (a correction does not un-leak anything).

    The outbox stores one email per line: "TO:<to> | <subject>", plus " | <body>" when there
    is a body. The body is saved as JSON text, so a multi-line message still fits on one line.
    """
    kind = COMPENSABLE

    def __init__(self, sandbox, to, subject, body=""):
        super().__init__(sandbox)
        self.to, self.subject, self.body = to, subject, body

    def goes_to_known_people(self):
        """A mailing-list name (no "@"), a company address, or a customer in the database right
        now. A customer DELETED earlier in the same task no longer counts (the safe side)."""
        to = self.to.strip().lower()
        if "@" not in to or to.endswith(COMPANY_DOMAIN):
            return True                                 # a list name, or our own company
        con = self.sb.db()
        try:
            found = con.execute("SELECT 1 FROM customers WHERE lower(email) = ?",
                                (to,)).fetchone()
        except sqlite3.OperationalError:                # the customers table was dropped
            found = None
        finally:
            con.close()
        return found is not None

    def describe(self):
        return f"send email to '{self.to}': {self.subject}"

    def do(self):
        line = f"TO:{self.to} | {self.subject}"
        if self.body:
            line += f" | {json.dumps(self.body, ensure_ascii=False)}"
        self.line = line                    # kept, so erase() can find it again
        with open(self.sb.outbox_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def undo(self):
        # A correction only helps if it arrives BEFORE the email was read (Step I2's outside
        # world). The runtime sets `late` when it was already read; that correction is written
        # differently, so it does not count as undoing anything.
        if getattr(self, "late", False):
            note = f"LATE CORRECTION, already read: please ignore '{self.subject}'"
        else:
            note = f"CORRECTION: please ignore '{self.subject}'"
        with open(self.sb.outbox_path, "a", encoding="utf-8") as f:
            f.write(f"TO:{self.to} | {note}\n")

    def erase(self):
        """Step I2: the email was still HELD (never delivered), so remove it from the outbox
        completely, with no correction: nobody ever saw it."""
        lines = self.sb.outbox_lines()
        if self.line in lines:
            del lines[len(lines) - 1 - lines[::-1].index(self.line)]      # the last copy
        with open(self.sb.outbox_path, "w", encoding="utf-8") as f:
            f.writelines(line + "\n" for line in lines)


def read_outbox_line(line):
    """'TO:a | subject | "body"' -> (address, subject, body). The reverse of SendEmail.do().
    The agent may not put "|" in an address or subject, so splitting at the first two
    separators is safe."""
    parts = line.removeprefix("TO:").split(" | ", 2)
    address, subject = parts[0], parts[1] if len(parts) > 1 else ""
    body = ""
    if len(parts) > 2:
        try:
            body = json.loads(parts[2])
        except ValueError:
            body = parts[2]
    return address, subject, body


# ---------------- IRREVERSIBLE ----------------

class TransferMoney(Action):
    """Once money leaves, it is gone. Only a human can allow this. The exception (Step I1):
    money moved to another of the company's OWN accounts can be moved back, so the classifier
    makes that call COMPENSABLE, and undo() moves it back."""
    kind = IRREVERSIBLE

    def __init__(self, sandbox, amount, to="supplier"):
        super().__init__(sandbox)
        self.amount, self.to = amount, to
        self.new_account = False

    def describe(self):
        return f"transfer {self.amount} to '{self.to}'"

    def prepare(self):
        # Remember if this transfer CREATES the account, so moving the money back can remove it
        # again (found by undo_conformance.py, Step I4: an empty account was left behind).
        self.new_account = self.to not in self.sb.bank()

    def do(self):
        bank = self.sb.bank()
        bank["company"] -= self.amount
        bank[self.to] = bank.get(self.to, 0) + self.amount
        self.sb.save_bank(bank)

    def undo(self):
        if self.to not in COMPANY_ACCOUNTS:
            raise RuntimeError("Money sent outside the company cannot be undone")
        self.move_back()                            # a compensating transfer back

    def erase(self):
        """Step I2: the transfer was still HELD (never settled), so it is simply reversed."""
        self.move_back()

    def move_back(self):
        bank = self.sb.bank()
        bank[self.to] -= self.amount
        bank["company"] += self.amount
        if self.new_account and bank[self.to] == 0:
            del bank[self.to]                       # the account did not exist before
        self.sb.save_bank(bank)


class DropTable(Action):
    """DROP TABLE with no backup policy. Treated as irreversible."""
    kind = IRREVERSIBLE

    def describe(self):
        return "DROP TABLE customers"

    def do(self):
        con = self.sb.db()
        con.execute("DROP TABLE customers")
        con.commit()
        con.close()

    def undo(self):
        raise RuntimeError("Irreversible action cannot be undone")


# ---------------- saving an action to the journal and back (Step 7) ----------------
# Crash recovery needs to rebuild an action in a FRESH process to call its undo(). Every action
# keeps its undo info in plain fields (old_content, trash_copy, saved_rows, old_emails,
# before_max_id), all JSON-friendly, so we can store the field values and restore them later.

ACTION_TYPES = {cls.__name__: cls for cls in
                (WriteFile, DeleteFile, DeleteRows, AddCustomer, SetEmail, SendEmail,
                 TransferMoney, DropTable)}


def to_journal(action):
    """An action -> a plain dict {type, kind, state} safe to write as JSON. `state` is every
    field except the sandbox (which the recovering process supplies)."""
    state = {key: value for key, value in vars(action).items() if key != "sb"}
    return {"type": type(action).__name__, "kind": action.kind, "state": state}


def from_journal(sandbox, record):
    """Rebuild the action a to_journal() record describes, attached to `sandbox`. __new__ skips
    __init__, so a DeleteFile keeps its ORIGINAL trash path instead of making a new one."""
    cls = ACTION_TYPES[record["type"]]
    action = cls.__new__(cls)
    action.sb = sandbox
    action.__dict__.update(record["state"])
    return action
