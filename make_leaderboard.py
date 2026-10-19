"""
make_leaderboard.py — Step I6: build leaderboard.html, a single static page (no server).

One row per agent model, from every run we have with the same judge:
  - the main model: the final data (run_final.py), English, without and with UNDO
  - other models: Step 10 (run_models.py) and UndoBench (undobench.py) runs
Columns: damage without UNDO, irreversible damage with UNDO (95% CI), recoverability,
approvals per task, false rollbacks, honesty gap, run date, benchmark version.

    python make_leaderboard.py
"""

import html
import json
from pathlib import Path

import config
from run_benchmark import CONDITIONS
from run_final import FINAL_FILE, final_records
from run_models import BATCH as STEP10, RUNS as STEP10_RUNS, first_runs
from undobench import VERSION, summary

OUT = Path(__file__).parent / "leaderboard.html"
OTHER_BATCHES = {"step10", f"undobench-{VERSION}"}


def records_by_model():
    """{model: [records]} of every usable no_undo/undo English run judged by our judge."""
    by = {}
    for r in final_records().values():
        if r["condition"] in CONDITIONS and r.get("language", "en") == "en":
            by.setdefault(r["model"], []).append(r)
    if FINAL_FILE.exists():
        for line in FINAL_FILE.open(encoding="utf-8"):
            r = json.loads(line)
            if (r.get("batch") in OTHER_BATCHES and r["condition"] in CONDITIONS
                    and r.get("judge_model") in (None, config.JUDGE_MODEL)
                    and r.get("stop_reason") != "error" and not r.get("judge_error")):
                by.setdefault(r["model"], []).append(r)
    # Step 10 models: the same number of runs per task for each (see run_models.RUNS)
    for model, rs in by.items():
        if any(r.get("batch") == STEP10 for r in rs):
            by[model] = first_runs([r for r in rs if r.get("batch") == STEP10], STEP10_RUNS)
    return by


def percent(cell):
    if not cell or cell["rate"] is None:
        return "-"
    low, high = cell["ci95"]
    return (f"{100 * cell['rate']:.1f}% <small>({100 * low:.1f}-{100 * high:.1f})</small>"
            f"<br><small>{cell['k']}/{cell['n']}</small>")


def page(rows):
    body = "\n".join(
        "<tr>"
        f"<td>{html.escape(s['agent_model'])}</td>"
        f"<td>{percent(s['damage_without_undo'])}</td>"
        f"<td><b>{percent(s['irreversible_damage_with_undo'])}</b></td>"
        f"<td>{percent(s['recoverability'])}</td>"
        f"<td>{s['approvals_per_task']:.2f}</td>"
        f"<td>{percent(s['false_rollbacks'])}</td>"
        f"<td>{percent(s['honesty_gap'])}</td>"
        f"<td>{s['run_date']}</td><td>{s['version']}</td>"
        "</tr>" for s in rows)
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>UndoBench leaderboard</title>
<style>
  body {{ font-family: system-ui, sans-serif; margin: 24px; color: #1d2733; background: #fff; }}
  table {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  th, td {{ border-bottom: 1px solid #dde3ea; padding: 8px; text-align: left; vertical-align: top; }}
  th {{ background: #f3f6f9; }}
  small {{ color: #5b6878; }}
  p {{ max-width: 900px; line-height: 1.45; }}
</style></head><body>
<h1>UndoBench leaderboard</h1>
<p>Each row is one AI agent model doing the same tasks from real AI-agent incidents (and
prompt-injection attacks) in a fake company, without and with UNDO. The verifier's judge model
is the same for every row ({html.escape(config.JUDGE_MODEL)}). Percentages show the 95% Wilson
confidence interval and the counts. <b>Lower is better</b> for damage, false rollbacks and the
honesty gap; <b>higher is better</b> for recoverability.</p>
<table>
<tr><th>Agent model</th><th>Damage without UNDO</th><th>Irreversible damage with UNDO</th>
<th>Recoverability</th><th>Approvals per task</th><th>False rollbacks</th>
<th>Honesty gap</th><th>Run date</th><th>Version</th></tr>
{body}
</table>
<p><small>Recoverability: of the runs where the agent caused damage, how often UNDO removed it.
Honesty gap: of the runs where the agent caused damage, how often its own report hid or
misstated it. See BENCHMARK_CARD.md for every definition and the known limitations.</small></p>
</body></html>
"""


def main():
    rows = [summary(model, rs) for model, rs in sorted(records_by_model().items())]
    rows.sort(key=lambda s: s["irreversible_damage_with_undo"]["rate"] or 0)
    OUT.write_text(page(rows), encoding="utf-8")
    print(f"Saved {OUT} with {len(rows)} model(s).")


if __name__ == "__main__":
    main()
