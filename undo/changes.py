"""
changes.py — a short, readable summary of what the agent changed (Step 3).

UNDO's Step 3 verifiers never see the scenario or its answer key. They see only two things:
the user's task text, and this summary of how the world now differs from before the task.

summarize(before, after) compares two snapshots (see Sandbox.snapshot) and returns a list of
changes. Each change is a dict with:
    kind  - what sort of change, e.g. "file_deleted" (the rule verifier reads this)
    text  - one plain-English line, e.g. "file deleted: cache.tmp (12 chars) ..." (the LLM
            judge reads these lines)
plus the details a rule needs, such as the file name or how many rows went.
Read-only tool calls change nothing, so they never appear here.
"""

import json

from .actions import read_outbox_line

PREVIEW_CHARS = 200          # how much of a file's text the summary shows
EMAIL_PREVIEW_CHARS = 1500   # emails get more: an email's words ARE its effect, and a harmful
                             # promise can hide at the end of a polite letter
MAX_ROWS_LISTED = 10         # how many customer rows one line lists before it says "and N more"


def preview(text, limit=PREVIEW_CHARS):
    """The text in quotes, cut short if it is long. json.dumps writes a newline as \\n, so a
    preview always stays on one line, and the quotes show exactly where the text starts and
    ends (an empty file shows as "")."""
    more = f" ...(+{len(text) - limit} more chars)" if len(text) > limit else ""
    return json.dumps(text[:limit], ensure_ascii=False) + more


def change(kind, text, **details):
    return dict(kind=kind, text=text, **details)


# ---------- files ----------

def file_changes(before, after):
    changes = []
    for name in sorted(set(before["files"]) | set(after["files"])):
        old, new = before["files"].get(name), after["files"].get(name)
        if old == new:
            continue
        if old is None:
            changes.append(change("file_created",
                                  f"file created: {name} ({len(new)} chars): {preview(new)}",
                                  name=name))
        elif new is None:
            changes.append(change("file_deleted",
                                  f"file deleted: {name} ({len(old)} chars), it held: {preview(old)}",
                                  name=name))
        else:
            changes.append(change("file_changed",
                                  f"file rewritten: {name} ({len(old)} -> {len(new)} chars), "
                                  f"was: {preview(old)}, now: {preview(new)}",
                                  name=name))
    return changes


# ---------- customers ----------

def list_rows(rows):
    """'5 | Test User | test@test.com; 6 | QA Bot | qa@test.com', cut short if there are many."""
    shown = "; ".join(" | ".join(str(x) for x in row) for row in rows[:MAX_ROWS_LISTED])
    if len(rows) > MAX_ROWS_LISTED:
        shown += f"; and {len(rows) - MAX_ROWS_LISTED} more"
    return shown


def customer_changes(before, after):
    old = {row[0]: row for row in before["customers"] or []}    # id -> (id, name, email)
    total = len(old)
    if after["customers"] is None:
        if before["customers"] is None:
            return []                       # there was no table before either
        return [change("table_dropped",
                       f"customers: the whole table was dropped ({total} rows)",
                       count=total, total=total, ids=sorted(old))]

    new = {row[0]: row for row in after["customers"]}
    deleted = [old[i] for i in old if i not in new]
    added = [new[i] for i in new if i not in old]
    edited = [(old[i], new[i]) for i in old if i in new and old[i] != new[i]]
    changes = []
    if deleted:
        changes.append(change("rows_deleted",
                              f"customers: {len(deleted)} of {total} rows deleted: {list_rows(deleted)}",
                              count=len(deleted), total=total, ids=[row[0] for row in deleted]))
    if edited:
        pairs = "; ".join(f"{list_rows([a])} -> {list_rows([b])}" for a, b in edited[:MAX_ROWS_LISTED])
        if len(edited) > MAX_ROWS_LISTED:
            pairs += f"; and {len(edited) - MAX_ROWS_LISTED} more"
        changes.append(change("rows_changed",
                              f"customers: {len(edited)} of {total} rows changed: {pairs}",
                              count=len(edited), total=total, ids=[a[0] for a, _ in edited]))
    if added:
        changes.append(change("rows_added",
                              f"customers: {len(added)} row(s) added: {list_rows(added)}",
                              count=len(added), total=total, ids=[row[0] for row in added]))
    return changes


# ---------- email ----------

def email_changes(before, after):
    # The outbox only ever grows (sending appends a line), so the new emails are at the end.
    changes = []
    for line in after["outbox"][len(before["outbox"]):]:
        to, subject, body = read_outbox_line(line)
        text = f"email sent to {to}, subject: {preview(subject)}"
        if body:
            text += f", body: {preview(body, EMAIL_PREVIEW_CHARS)}"
        changes.append(change("email_sent", text, to=to))
    return changes


# ---------- bank ----------

def amount_text(x):
    """5000.0 -> '5000', 12.5 -> '12.5' (no long decimals in the summary)."""
    return f"{x:.2f}".rstrip("0").rstrip(".")


def bank_changes(before, after):
    changes = []
    for account in sorted(set(before["bank"]) | set(after["bank"])):
        old, new = before["bank"].get(account, 0), after["bank"].get(account, 0)
        if old != new:
            difference = new - old
            sign = "+" if difference > 0 else ""
            changes.append(change("money_moved",
                                  f"bank: {account} {amount_text(old)} -> {amount_text(new)} "
                                  f"({sign}{amount_text(difference)})",
                                  account=account, difference=difference))
    return changes


# ---------- everything ----------

def summarize(before, after):
    """Every change between two snapshots: files, then customers, then email, then money."""
    return (file_changes(before, after) + customer_changes(before, after)
            + email_changes(before, after) + bank_changes(before, after))


def as_text(changes):
    """The summary as the lines the LLM judge reads."""
    return "\n".join(c["text"] for c in changes) or "nothing changed"
