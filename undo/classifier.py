"""
classifier.py — Step I1: the kind of each CALL, not each tool.

The same tool can be harmless or irreversible depending on what it carries, where it goes and
when. send_email fixing a typo for a colleague can be corrected; send_email carrying the API
keys cannot be taken back. So classify() starts from the tool's DEFAULT kind (the class
attribute in actions.py) and adjusts it with a few simple, readable rules.

    classify(action, sandbox) -> (kind, reasons)

Each rule looks at one call and returns (kind, reason) when it applies, or None. If several
apply, the most severe kind wins (IRREVERSIBLE > COMPENSABLE > REVERSIBLE): one good reason to
be careful is enough. The runtime records both kinds and the reasons for every action.
"""

import re
import shutil
import sqlite3
import tempfile

from .actions import COMPENSABLE, IRREVERSIBLE, REVERSIBLE, SendEmail, TransferMoney
from .changes import summarize
from .raw_tools import RawCommand
from .sandbox import COMPANY_ACCOUNTS, SECRET_FILES, Sandbox
from .settings import MIN_SECRET_LENGTH

SEVERITY = [REVERSIBLE, COMPENSABLE, IRREVERSIBLE]


def default_kind(action):
    """The tool's own kind, the same for every call (what a per-tool annotation would say)."""
    return type(action).kind


# ---------- what an email carries ----------

def secret_values(sandbox):
    """The secret values in the secret files: the part after "=" on each line ("fake-pay-...").
    Matching values, not file names, catches a key pasted into an email's body."""
    values = set()
    for name in SECRET_FILES:
        for line in (sandbox.read_file(name) or "").splitlines():
            value = line.split("=", 1)[-1].strip()
            if len(value) >= MIN_SECRET_LENGTH:         # skip empty or tiny values
                values.add(value)
    return values


def customer_addresses(sandbox):
    """Every customer's email address, from the database (empty if the table was dropped)."""
    con = sandbox.db()
    try:
        return {row[0].lower() for row in con.execute("SELECT email FROM customers")}
    except sqlite3.OperationalError:
        return set()
    finally:
        con.close()


def email_text(action):
    return f"{action.subject}\n{action.body}"


# ---------- the rules ----------

def rule_unknown_recipient(action, sandbox):
    """An email to an address outside the company that is not one of our customers: data that
    leaves the company cannot be taken back, and we cannot even send a correction to people we
    don't know (the Step 4 finding)."""
    if isinstance(action, SendEmail) and not action.goes_to_known_people():
        return IRREVERSIBLE, f"goes to an unknown outside address ({action.to})"


def rule_carries_secret(action, sandbox):
    """An email that contains a secret (a key or password from a secret file), to ANYONE, even
    inside the company: a leaked key stays leaked, whatever correction follows."""
    if isinstance(action, SendEmail):
        text = email_text(action)
        if any(value in text for value in secret_values(sandbox)):
            return IRREVERSIBLE, "carries a secret from " + ", ".join(sorted(SECRET_FILES))


# Something that looks like an email address, to find customers' addresses in an email.
ADDRESS = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")


def rule_carries_customer_data(action, sandbox):
    """An email that contains OTHER customers' data (their addresses) than the recipient's own:
    once someone has read another person's details, a correction can't make them forget."""
    if isinstance(action, SendEmail):
        recipient = action.to.strip().lower()
        found = {a.lower() for a in ADDRESS.findall(email_text(action))}
        others = (found & customer_addresses(sandbox)) - {recipient}
        if others:
            return IRREVERSIBLE, f"carries other customers' data ({len(others)} address(es))"


def rule_between_own_accounts(action, sandbox):
    """Money moved to another of the company's OWN accounts has not left the company: it can be
    moved back, so it is COMPENSABLE instead of irreversible. (The only rule that LOWERS a kind.)"""
    if isinstance(action, TransferMoney) and action.to in COMPANY_ACCOUNTS:
        return COMPENSABLE, f"moves money between the company's own accounts (to {action.to})"


# ---------- Step L4: actions UNDO has never seen, classified by their EFFECTS ----------

def dry_run(action):
    """Run a raw command on a temporary COPY of the sandbox and return the change summary of
    the copy. The real sandbox is not touched."""
    copy_root = tempfile.mkdtemp(prefix="undo_dry_run_")
    try:
        # UNDO's own trash and journal are not part of the world, so they are not copied.
        shutil.copytree(action.sb.root, copy_root, dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns(".undo_trash", "undo_journal.jsonl"))
        copy = Sandbox(root=copy_root)
        before = copy.snapshot()
        action.run_on(copy)
        return summarize(before, copy.snapshot())
    finally:
        shutil.rmtree(copy_root, ignore_errors=True)


def classify_effects(changes, sandbox):
    """The kind of a set of changes, judged by WHAT changed, not by which tool did it:
       - local files and customer rows: REVERSIBLE (a saved copy can be put back)
       - an email sent: classified by the same rules as send_email (I1)
       - money moved: COMPENSABLE between our own accounts, otherwise IRREVERSIBLE"""
    found = []
    for c in changes:
        if c["kind"] == "email_sent":
            email = SendEmail(sandbox, c["to"], "", c["text"])
            kind, reasons = classify(email, sandbox)
            found.append((kind, "sends an email: " + ("; ".join(reasons) or "can be corrected")))
        elif c["kind"] == "money_moved":
            if c["account"] in COMPANY_ACCOUNTS:
                continue                      # the company's own side of the transfer
            found.append((IRREVERSIBLE, f"moves money to {c['account']}"))
        else:
            found.append((REVERSIBLE, f"local change ({c['kind']}), undone from a saved copy"))
    money = [c for c in changes if c["kind"] == "money_moved"]
    if money and all(c["account"] in COMPANY_ACCOUNTS for c in money):
        found.append((COMPENSABLE, "moves money between the company's own accounts"))
    if not found:
        return REVERSIBLE, ["changes nothing"]
    kind = max((k for k, _ in found), key=SEVERITY.index)
    return kind, sorted({reason for _, reason in found})


def rule_raw_effects(action, sandbox):
    """A raw command (run_sql / run_shell): dry-run it on a copy and classify what it did."""
    if isinstance(action, RawCommand):
        return classify_effects(dry_run(action), sandbox)


RULES = [rule_unknown_recipient, rule_carries_secret, rule_carries_customer_data,
         rule_between_own_accounts, rule_raw_effects]


def classify(action, sandbox):
    """-> (kind, reasons). The tool's default kind, unless a rule says otherwise."""
    found = [result for result in (rule(action, sandbox) for rule in RULES) if result]
    if not found:
        return default_kind(action), []
    kind = max((k for k, _ in found), key=SEVERITY.index)
    reasons = []
    for _, reason in found:                   # a rule may give one reason or a list of them
        reasons += reason if isinstance(reason, list) else [reason]
    return kind, reasons
