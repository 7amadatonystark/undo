"""
undobench.py — Step I6: run UndoBench on ANY OpenAI-compatible model endpoint, in one command.

    python undobench.py --model some/model --base-url https://host/v1 --key-env MY_API_KEY

It runs every scenario (the incident benchmark and the prompt-injection attacks) without and
with UNDO, RUNS times each, and writes:
  - every run, one JSON line each, to results/runs.jsonl (batch "undobench-<version>")
  - a summary in the format documented in BENCHMARK_CARD.md, to results/undobench_<model>.json

The judge (the verifier's AI) stays the same for every model on the leaderboard, so models are
compared fairly: by default it is JUDGE_MODEL on NVIDIA's API (needs NVIDIA_API_KEY in .env).
Use --judge-model / --judge-base-url / --judge-key-env to put it elsewhere.

The key is read from an environment variable (never typed on the command line or saved).
Runs are resumable: runs already saved are skipped.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import openai

import config
import llm
from analyze import damaged, misreported, wilson
from injection_scenarios import INJECTION_SCENARIOS
from run_benchmark import BLOCKED_REASON, CONDITIONS, run_one
from run_final import FINAL_FILE, save_shared
from scenarios import SCENARIOS

VERSION = "1.0"
BATCH = f"undobench-{VERSION}"
RUNS = 3


def connect(model, base_url, key_env):
    """Send this model's calls to its own endpoint (no change if base_url is empty)."""
    if base_url:
        key = os.getenv(key_env, "")
        if not key:
            raise SystemExit(f"Set the environment variable {key_env} to the endpoint's API key.")
        llm.OTHER_CLIENTS[model] = openai.OpenAI(base_url=base_url, api_key=key, max_retries=0,
                                                 timeout=120)


def saved(model):
    """{(scenario, condition, run): record} of this model's UndoBench runs."""
    out = {}
    if FINAL_FILE.exists():
        for line in FINAL_FILE.open(encoding="utf-8"):
            r = json.loads(line)
            if (r.get("batch") == BATCH and r.get("model") == model
                    and r.get("judge_model") in (None, config.JUDGE_MODEL)
                    and r.get("stop_reason") != "error" and not r.get("judge_error")):
                out[(r["scenario"], r["condition"], r["run"])] = r
    return out


def rate(k, n):
    low, high = wilson(k, n)
    return {"k": k, "n": n, "rate": k / n if n else None, "ci95": [low, high]}


def summary(model, records):
    """The documented summary (see BENCHMARK_CARD.md, 'Result format')."""
    no = [r for r in records if r["condition"] == "no_undo"]
    un = [r for r in records if r["condition"] == "undo"]
    damaging = [r for r in un if r["damaged_before"]]
    harmless = [r for r in un if not r["damaged_before"] and BLOCKED_REASON not in r["reasons"]]
    reported = [r for r in no + un if r["damaged_before"] and (r.get("honesty") or {}).get("label")]
    return {
        "benchmark": "UndoBench", "version": VERSION, "agent_model": model,
        "judge_model": config.JUDGE_MODEL, "run_date": max(r["timestamp"] for r in records)[:10],
        "scenarios": len({r["scenario"] for r in records}),
        "runs": len(records),
        "damage_without_undo": rate(damaged(no), len(no)),
        "irreversible_damage_with_undo": rate(damaged(un), len(un)),
        "recoverability": rate(sum(not r["damaged"] for r in damaging), len(damaging)),
        "approvals_per_task": sum(r["approvals_asked"] for r in un) / len(un) if un else None,
        "false_rollbacks": rate(sum(r["decision"] == "rollback" for r in harmless), len(harmless)),
        "honesty_gap": rate(sum(misreported(r) for r in reported), len(reported)),
    }


def main():
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Run UndoBench on any OpenAI-compatible endpoint.")
    parser.add_argument("--model", required=True, help="the agent model's name at the endpoint")
    parser.add_argument("--base-url", help="the endpoint (default: NVIDIA's API)")
    parser.add_argument("--key-env", default="UNDOBENCH_API_KEY",
                        help="environment variable holding the endpoint's key")
    parser.add_argument("--judge-model", default=config.JUDGE_MODEL)
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-key-env", default="UNDOBENCH_JUDGE_API_KEY")
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()

    connect(args.model, args.base_url, args.key_env)
    connect(args.judge_model, args.judge_base_url, args.judge_key_env)
    config.AGENT_MODEL, config.JUDGE_MODEL = args.model, args.judge_model

    scenarios = SCENARIOS + INJECTION_SCENARIOS
    done = saved(args.model)
    todo = [(s, c, n) for n in range(1, args.runs + 1) for s in scenarios for c in CONDITIONS
            if (s["name"], c, n) not in done]
    print(f"UndoBench {VERSION}: {args.model}, judge {args.judge_model}. {len(todo)} run(s) to do, "
          f"about {len(todo) * 8} API calls.")
    for i, (scenario, condition, n) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {scenario['name']} [{condition}] run {n}", flush=True)
        save_shared(run_one(scenario, condition, run_number=n, batch=BATCH))

    records = list(saved(args.model).values())
    result = summary(args.model, records)
    out = Path(__file__).parent / "results" / f"undobench_{args.model.replace('/', '_')}.json"
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
