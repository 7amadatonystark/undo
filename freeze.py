"""
freeze.py — Step 6: freeze every setting before the final runs.

Writes FROZEN.md: today's date, every number in undo/settings.py, the agent and judge models,
the temperature and the final batch name. run_final.py refuses to run if anything differs
from FROZEN.md, so nothing can change in the middle of the final data. If something MUST
change, write down why in FROZEN.md's "Changes" section, run freeze.py again, and rerun
everything.

    python freeze.py
"""

import json
from datetime import date
from pathlib import Path

import config
import undo.settings as settings

FROZEN = Path(__file__).parent / "FROZEN.md"


def current():
    """Everything that is frozen, as a plain dict."""
    from run_final import BATCH
    values = {name: value for name, value in vars(settings).items() if name.isupper()}
    return {"AGENT_MODEL": config.AGENT_MODEL, "JUDGE_MODEL": config.JUDGE_MODEL,
            "FINAL_BATCH": BATCH, **values}


def frozen():
    """The settings written in FROZEN.md, or None if it doesn't exist."""
    if not FROZEN.exists():
        return None
    text = FROZEN.read_text(encoding="utf-8")
    block = text.split("```json", 1)[1].split("```", 1)[0]
    return json.loads(block)


def differences():
    """[(name, frozen value, current value)] for everything that changed since the freeze."""
    was, now = frozen() or {}, current()
    return [(k, was.get(k), now.get(k)) for k in sorted(set(was) | set(now)) if was.get(k) != now.get(k)]


def main():
    FROZEN.write_text(
        "# Frozen settings for the final data (Step 6)\n\n"
        f"Frozen on {date.today().isoformat()}. After this, nothing below may change. If something must\n"
        "change, write down what and why under Changes, run `python freeze.py` again, and rerun\n"
        "every final run (a new batch name).\n\n"
        "```json\n" + json.dumps(current(), indent=2) + "\n```\n\n"
        "## Changes\n\n(none)\n", encoding="utf-8")
    print(f"Wrote {FROZEN.name}:")
    print(json.dumps(current(), indent=2))


if __name__ == "__main__":
    main()
