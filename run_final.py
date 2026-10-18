"""
run_final.py — Step 6: collect the FINAL data (everything before this was practice).

What it runs (each job = one scenario, one condition, one language, one run number):
  English: every benchmark scenario (scenarios.py) and every injection scenario
           (injection_scenarios.py), in all 5 conditions: no_undo, undo, and the three
           ablations (no_gate, no_verifier, no_rollback), RUNS times each.
  Arabic:  every translated scenario (tasks_ar.json), in msa and egy, without and with UNDO,
           RUNS times each. The English runs above are the comparison.

Every run is one JSON line in results/runs.jsonl (batch "final3"). It is
resumable: runs already saved are skipped, so it can stop (rate limit, lost connection, laptop
closed) and continue later. Jobs go in run-number order (every scenario's run 1, then every
run 2, ...), so if we must stop early, every scenario has the same number of runs so far.

UNDO's decisions use the combined verifier, as in Steps 4 and 5.

    python run_final.py --estimate      # count the runs and API calls, run nothing
    python run_final.py                 # run everything not yet saved (takes hours)
    python run_final.py --part arabic   # only the Arabic part (or: english)
    python run_final.py --worker 2/3    # run share 2 of 3; start 1/3, 2/3 and 3/3 side by side

Why workers? A run takes about a minute, mostly waiting for the model to answer, so one
process uses only ~9 of our 30 requests per minute. With --worker i/n, each of n processes
takes every n-th job of the SAME fixed list and is throttled to 30/n requests per minute, so
together they still never pass 30 per minute (our API limit), but finish about n times sooner.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import config
import llm
from agent import MAX_TOOL_CALLS, REPORT_MAX_LOOKS
from injection_scenarios import INJECTION_SCENARIOS
from run_benchmark import ALL_CONDITIONS, CONDITIONS, DEFAULT_VERIFIER, run_one, save
from scenarios import SCENARIOS, in_language, translated

FINAL_FILE = Path(__file__).parent / "results" / "runs.jsonl"
# "final" was the first try: its first 25 runs had a bug in the honesty check (look-only
# results cut to one line, Step 8). Those lines stay in the file but no longer count.
# "final2" (35 usable runs) was stopped: the old agent model was throttled by NVIDIA, and the
# honesty judge counted tiny details (a trailing newline) as false claims. "final3" uses the
# new agent model and the fixed honesty instructions.
BATCH = "final3"
RUNS = 5
VERIFIER = DEFAULT_VERIFIER
ARABIC = ["msa", "egy"]
# API calls per run: at most the agent's limit + 1, the report (1 + its looks), the honesty
# judge and the verifier. "Typical" is from the Step 4-5 practice runs (about 5 tool calls).
MOST_CALLS_PER_RUN = MAX_TOOL_CALLS + 1 + 1 + REPORT_MAX_LOOKS + 1 + 1
TYPICAL_CALLS_PER_RUN = 8


def all_scenarios():
    return SCENARIOS + INJECTION_SCENARIOS


def jobs(part=None):
    """Every (scenario, condition, language, run) the final data needs, run 1 first."""
    planned = []
    for run in range(1, RUNS + 1):
        if part in (None, "english"):
            planned += [(s, c, "en", run) for s in all_scenarios() for c in ALL_CONDITIONS]
        if part in (None, "arabic"):
            planned += [(s, c, lang, run) for lang in ARABIC for s in all_scenarios()
                        if translated(s["name"], lang) for c in CONDITIONS]
    return planned


def key(scenario_name, condition, language, run):
    return (scenario_name, condition, language, run)


def final_records():
    """{key: record} for every final run that counts: this batch and agent model, a scenario
    that still exists with the same task wording in its language, the same verifier and judge
    (for runs with UNDO), and no API error (those are redone). The latest copy of a run wins."""
    tasks = {(s["name"], lang): in_language(s, lang)["task"]
             for s in all_scenarios() for lang in ["en"] + ARABIC if translated(s["name"], lang)}
    latest = {}
    if FINAL_FILE.exists():
        for line in FINAL_FILE.open(encoding="utf-8"):
            r = json.loads(line)
            language = r.get("language", "en")
            if (r.get("batch") == BATCH and r.get("model") == config.AGENT_MODEL
                    and tasks.get((r["scenario"], language)) == r.get("task")
                    and (r["condition"] == "no_undo"
                         or (r.get("verifier") == VERIFIER
                             and r.get("judge_model") == config.JUDGE_MODEL))
                    and r.get("stop_reason") != "error" and not r.get("judge_error")):
                latest[key(r["scenario"], r["condition"], language, r["run"])] = r
    return latest


def save_shared(record):
    """Append one run while other workers may be appending too. A lock file lets only one
    process write at a time, so two lines can never be mixed together. A lock left behind by a
    crashed worker is removed after 30 seconds."""
    lock = FINAL_FILE.with_suffix(".lock")
    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL)
            break
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > 30:
                    lock.unlink()
            except FileNotFoundError:
                pass
            time.sleep(0.05)
    try:
        save(record, FINAL_FILE)
    finally:
        os.close(handle)
        lock.unlink()


def main():
    sys.stdout.reconfigure(errors="replace")      # never crash on Arabic in the terminal
    parser = argparse.ArgumentParser()
    parser.add_argument("--estimate", action="store_true", help="only count runs and API calls")
    parser.add_argument("--part", choices=["english", "arabic"], help="run only one part")
    parser.add_argument("--worker", default="1/1", metavar="I/N",
                        help="run share I of N side by side (each gets 30/N requests a minute)")
    args = parser.parse_args()
    i, n = map(int, args.worker.split("/"))
    llm.SECONDS_BETWEEN_REQUESTS = 60 / (llm.MAX_REQUESTS_PER_MINUTE / n)

    done = final_records()
    mine = jobs(args.part)[i - 1::n]           # a fixed share, so workers never overlap
    todo = [j for j in mine if key(j[0]["name"], j[1], j[2], j[3]) not in done]
    print(f"Final data: batch {BATCH!r}, agent {config.AGENT_MODEL}, judge {config.JUDGE_MODEL}, "
          f"verifier {VERIFIER}, {RUNS} runs each.")
    print(f"Worker {i}/{n}: {len(mine)} of {len(jobs(args.part))} runs are this worker's, "
          f"{len(mine) - len(todo)} saved, {len(todo)} to do "
          f"({llm.MAX_REQUESTS_PER_MINUTE / n:.0f} requests/min).")
    rate = llm.MAX_REQUESTS_PER_MINUTE / n
    print(f"API calls: about {len(todo) * TYPICAL_CALLS_PER_RUN} typical "
          f"(~{len(todo) * TYPICAL_CALLS_PER_RUN / rate / 60:.1f} h at {rate:.0f}/min), "
          f"at most {len(todo) * MOST_CALLS_PER_RUN} "
          f"(~{len(todo) * MOST_CALLS_PER_RUN / rate / 60:.1f} h). A run also waits ~1 min "
          f"for the model, so expect about {len(todo) / 60:.0f} h for this worker.")
    if args.estimate:
        return

    for i, (scenario, condition, language, run) in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {scenario['name']} [{condition}] {language} run {run}", flush=True)
        record = run_one(scenario, condition, run_number=run, batch=BATCH, verifier=VERIFIER,
                         language=language)
        save_shared(record)
        print(f"      outcome={record['outcome']} decision={record['decision']} "
              f"rolled_back={record['rolled_back']} stop={record['stop_reason']}", flush=True)


if __name__ == "__main__":
    main()
