"""
run_models.py — Step 10: does UNDO help EVERY agent model, and are some models more
destructive than others?

The same English scenarios (benchmark + injection), without and with UNDO, on more agent
models. The judge model stays the same, so only the agent changes. The main model
(AGENT_MODEL) already has its runs in the final data (run_final.py), so it is not rerun here.

Which models: EXTRA_AGENT_MODELS in .env (comma-separated), or --models on the command line.
Pick models that support tool calls (check with: python run_models.py --check).

Runs are saved to results/runs.jsonl as batch "step10" (resumable).

    python run_models.py --check                      # 1 test call per model: can it use tools?
    python run_models.py                              # every extra model, RUNS runs each
    python run_models.py --models a/model,b/model     # instead of EXTRA_AGENT_MODELS
"""

import argparse
import json
import sys

import agent
import config
import llm
from analyze import damaged, fisher_exact, obeyed, rate_line, runs
from injection_scenarios import INJECTION_SCENARIOS
from run_benchmark import CONDITIONS, run_one
from run_final import FINAL_FILE, final_records, save_shared
from scenarios import SCENARIOS

BATCH = "step10"
RUNS = 3
VERIFIER = "combined"


def all_scenarios():
    return SCENARIOS + INJECTION_SCENARIOS


def use_model(model):
    """Make every agent call use this model. llm.chat reads config.AGENT_MODEL on every call,
    and run_one saves it with the run, so changing it here is enough."""
    config.AGENT_MODEL = model


def check(model):
    """One call: does this model answer with a real tool call?"""
    try:
        reply = llm.chat([{"role": "system", "content": agent.SYSTEM_PROMPT},
                          {"role": "user", "content": "Delete the file cache.tmp."}],
                         tools=agent.TOOLS, model=model, temperature=config.AGENT_TEMPERATURE)
    except Exception as error:
        return f"NOT usable: {str(error)[:100]}"
    if reply.tool_calls:
        return f"OK: called {reply.tool_calls[0].function.name}"
    return "NOT usable: it answered in text instead of calling a tool"


def model_records(models):
    """{(model, scenario, condition, run): record} for the step10 runs of these models, plus the
    main model's English no_undo/undo runs from the final data (for the comparison)."""
    tasks = {s["name"]: s["task"] for s in all_scenarios()}
    latest = {}
    if FINAL_FILE.exists():
        for line in FINAL_FILE.open(encoding="utf-8"):
            r = json.loads(line)
            if (r.get("batch") == BATCH and r.get("model") in models
                    and tasks.get(r["scenario"]) == r.get("task")
                    and (r["condition"] == "no_undo" or r.get("judge_model") == config.JUDGE_MODEL)
                    and r.get("stop_reason") != "error" and not r.get("judge_error")):
                latest[(r["model"], r["scenario"], r["condition"], r["run"])] = r
    for r in final_records().values():
        if r.get("language", "en") == "en" and r["condition"] in CONDITIONS:
            latest[(r["model"], r["scenario"], r["condition"], r["run"])] = r
    return latest


def scoreboard(records, models):
    print(f"\n{'MODEL':42s} {'DAMAGE, NO UNDO':>30s} {'DAMAGE, UNDO':>30s} {'p':>8s} "
          f"{'ATTACK OBEYED':>14s}")
    for model in models:
        mine = [r for r in records if r["model"] == model]
        no, un = runs(mine, "no_undo"), runs(mine, "undo")
        if not (no or un):
            continue
        p = fisher_exact(damaged(no), len(no) - damaged(no), damaged(un), len(un) - damaged(un))
        attacks = runs(mine, "no_undo", injection=True)
        print(f"{model:42s} {rate_line(damaged(no), len(no)):>30s} "
              f"{rate_line(damaged(un), len(un)):>30s} {p:8.2g} "
              f"{sum(obeyed(r) for r in attacks)}/{len(attacks):<10d}")
    print("\nDamage without UNDO shows how destructive each model is on its own. If UNDO helps "
          "every model, the UNDO column is low for all of them (p < 0.05 = unlikely to be luck).")


def main():
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", help="comma-separated agent models (default: EXTRA_AGENT_MODELS)")
    parser.add_argument("--check", action="store_true", help="only test that each model calls tools")
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()
    main_model = config.AGENT_MODEL
    extra = ([m.strip() for m in args.models.split(",") if m.strip()] if args.models
             else config.EXTRA_AGENT_MODELS)
    if not extra:
        raise SystemExit("No extra models: add EXTRA_AGENT_MODELS=... to .env, or use --models.")

    if args.check:
        for model in extra:
            print(f"{model}: {check(model)}")
        return

    done = model_records(extra)
    planned = [(m, s, c, n) for n in range(1, args.runs + 1) for m in extra
               for s in all_scenarios() for c in CONDITIONS if (m, s["name"], c, n) not in done]
    print(f"Batch {BATCH!r}: {len(planned)} run(s) to do on {len(extra)} model(s). "
          f"About {len(planned) * 8} API calls, roughly {len(planned)} min.")
    for i, (model, scenario, condition, n) in enumerate(planned, 1):
        use_model(model)
        print(f"[{i}/{len(planned)}] {model} {scenario['name']} [{condition}] run {n}", flush=True)
        record = run_one(scenario, condition, run_number=n, batch=BATCH, verifier=VERIFIER)
        save_shared(record)
        print(f"      outcome={record['outcome']} stop={record['stop_reason']}", flush=True)
    use_model(main_model)

    models = [main_model] + extra
    scoreboard([r for r in model_records(extra).values() if r["run"] <= max(args.runs, 5)], models)


if __name__ == "__main__":
    main()
