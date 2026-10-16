"""
demo.py — the booth demo. YOU play the human approver at the keyboard.

The agent edits a file, sends an email, then tries to DROP the customer table.
Answer 'n' to block it and watch UNDO roll everything back.

    python demo.py
"""

from undo import Sandbox, UndoRuntime, Blocked
from undo.actions import WriteFile, SendEmail, DropTable


def ask_human(action):
    answer = input(f"       Approve '{action.describe()}'? [y/N]: ")
    return answer.strip().lower() == "y"


def show_world(sb, title):
    print(f"\n--- {title} ---")
    print("config.json :", sb.read_file("config.json").strip())
    try:
        print("customers   :", sb.customer_count(), "rows")
    except Exception:
        print("customers   : TABLE GONE")
    print("outbox      :", sb.outbox_lines() or "(empty)")
    print()


def main():
    sb = Sandbox()
    rt = UndoRuntime(approver=ask_human)
    show_world(sb, "BEFORE")

    plan = [
        WriteFile(sb, "config.json", '{"mode": "debug"}\n'),
        SendEmail(sb, "all_users", "Maintenance tonight"),
        DropTable(sb),
    ]
    try:
        for action in plan:
            rt.execute(action)
        rt.commit()
    except Blocked:
        rt.rollback()

    show_world(sb, "AFTER")
    sb.cleanup()


if __name__ == "__main__":
    main()
