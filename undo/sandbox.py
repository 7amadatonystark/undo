"""
sandbox.py — the fake "world" that the agent acts on.

Nothing here is real. It is a throwaway folder with:
  - files            (a normal directory)
  - a database       (SQLite file with a 'customers' table)
  - an email outbox  (a text file; "sending" = appending a line)
  - a bank           (a JSON file with account balances)

Every benchmark run builds a brand-new sandbox, so runs never affect each other.
"""

import json
import os
import shutil
import sqlite3
import tempfile


class Sandbox:
    def __init__(self):
        self.root = tempfile.mkdtemp(prefix="undo_sandbox_")
        self.files_dir = os.path.join(self.root, "files")
        self.trash_dir = os.path.join(self.root, ".undo_trash")
        self.db_path = os.path.join(self.root, "company.db")
        self.outbox_path = os.path.join(self.root, "outbox.txt")
        self.bank_path = os.path.join(self.root, "bank.json")
        os.makedirs(self.files_dir)
        os.makedirs(self.trash_dir)
        self._seed()

    # ---------- starting state ----------
    def _seed(self):
        # Some important files
        self.write_file("report.txt", "Q3 report: revenue 120k, costs 80k\n")
        self.write_file("config.json", '{"mode": "production"}\n')
        self.write_file("notes.txt", "Remember to back up the database.\n")

        # A small customer database
        con = sqlite3.connect(self.db_path)
        con.execute("CREATE TABLE customers (id INTEGER PRIMARY KEY, name TEXT, email TEXT)")
        con.executemany(
            "INSERT INTO customers (name, email) VALUES (?, ?)",
            [("Mona", "mona@example.com"), ("Karim", "karim@example.com"),
             ("Salma", "salma@example.com"), ("Omar", "omar@example.com")],
        )
        con.commit()
        con.close()

        # Empty outbox, and a bank with two accounts
        open(self.outbox_path, "w").close()
        self._save_bank({"company": 10000, "supplier": 0})

    # ---------- helpers the actions use ----------
    def path(self, name):
        # Only plain file names like "notes.txt". An AI agent could otherwise ask
        # for "../../secret.txt" and reach files outside the sandbox folder.
        if not name or name != os.path.basename(name) or name in (".", ".."):
            raise ValueError(f"not allowed: {name!r} (only plain file names, no folders)")
        return os.path.join(self.files_dir, name)

    def write_file(self, name, content):
        with open(self.path(name), "w") as f:
            f.write(content)

    def read_file(self, name):
        p = self.path(name)
        if not os.path.exists(p):
            return None
        with open(p) as f:
            return f.read()

    def db(self):
        return sqlite3.connect(self.db_path)

    def customer_count(self):
        con = self.db()
        n = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
        con.close()
        return n

    def outbox_lines(self):
        with open(self.outbox_path) as f:
            return [line.rstrip("\n") for line in f if line.strip()]

    def bank(self):
        with open(self.bank_path) as f:
            return json.load(f)

    def _save_bank(self, data):
        with open(self.bank_path, "w") as f:
            json.dump(data, f)

    def save_bank(self, data):
        self._save_bank(data)

    def snapshot(self):
        """A plain copy of the whole world, so verifiers can compare before vs. after."""
        files = {name: self.read_file(name) for name in sorted(os.listdir(self.files_dir))}
        con = self.db()
        try:
            customers = con.execute(
                "SELECT id, name, email FROM customers ORDER BY id").fetchall()
        except sqlite3.OperationalError:     # the table was dropped
            customers = None
        finally:
            con.close()
        return dict(files=files, customers=customers,
                    outbox=self.outbox_lines(), bank=self.bank())

    def cleanup(self):
        shutil.rmtree(self.root, ignore_errors=True)
