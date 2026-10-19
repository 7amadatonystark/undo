"""
run_final.py — Step 6: collect the FINAL data (everything before this was practice).

What it runs (each job = one scenario, one condition, one language, one run number):
  English: every benchmark scenario (scenarios.py) and every injection scenario
           (injection_scenarios.py), in all 5 conditions: no_undo, undo, and the three
           ablations (no_gate, no_verifier, no_rollback), RUNS times each.
  Arabic:  every translated scenario (tasks_ar.json), in msa and egy, without and with UNDO,
           RUNS times each. The English runs above are the comparison.

Every run is one JSON line in results/runs.jsonl (batch "final4"). It is
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
from run_benchmark import ALL_CONDITIONS, CONDITIONS, DEFAULT_VERIFIER, run_one, save, settings
from heldout_scenarios import HELDOUT_SCENARIOS
from scenarios import PAIR_SCENARIOS, SCENARIOS, in_language, translated

FINAL_FILE = Path(__file__).parent / "results" / "runs.jsonl"
# "final" was the first try: its first 25 runs had a bug in the honesty check (look-only
# results cut to one line, Step 8). Those lines stay in the file but no longer count.
# "final2" (35 usable runs) was stopped: the old agent model was throttled by NVIDIA, and the
# honesty judge counted tiny details (a trailing newline) as false claims. "final3" uses the
# new agent model and the fixed honesty instructions.
# "final4": the redo after Steps I1-I3 (per-call classifier, hold window, permission-only
# conditions), with every condition the analysis compares.
BATCH = "final4"
RUNS = 5
HELDOUT_RUNS = 10        # the sealed test set gets 10 runs per condition: it is the headline
# When a run fails on the API, the worker does NOT wait on it: it moves on and comes back to
# every failed run in a later pass. After this many failures in a row it pauses a little
# (probably an outage), so it doesn't rush through the whole list failing every run.
FAILS_BEFORE_PAUSE = 3
PAUSE_SECONDS = 120
MAX_PASSES = 10
# Ablations that only change what UNDO does AFTER the agent has finished. The agent's work is
# exactly the same as with full UNDO, so they are computed by replaying the full-UNDO runs
# (derive_ablations.py, no API calls) instead of being run again.
REPLAYED = ["no_verifier", "no_rollback", "no_hold"]
VERIFIER = DEFAULT_VERIFIER
ARABIC = ["msa", "egy"]
# API calls per run: at most the agent's limit + 1, the report (1 + its looks), the honesty
# judge and the verifier. "Typical" is from the Step 4-5 practice runs (about 5 tool calls).
MOST_CALLS_PER_RUN = MAX_TOOL_CALLS + 1 + 1 + REPORT_MAX_LOOKS + 1 + 1
TYPICAL_CALLS_PER_RUN = 8


def all_scenarios():
    return SCENARIOS + INJECTION_SCENARIOS


def jobs(part=None):
    """Every (scenario, condition, language, run) the final data needs, run 1 first. Runs 6-10
    of no UNDO and full UNDO (English) come last: Step 6 asks for them "if time allows"."""
    planned = []
    for run in range(1, RUNS + 1):
        if part in (None, "english"):
            planned += [(s, c, "en", run) for s in all_scenarios() for c in ALL_CONDITIONS
                        if c not in REPLAYED]
        if part in (None, "arabic"):
            planned += [(s, c, lang, run) for lang in ARABIC for s in all_scenarios()
                        if translated(s["name"], lang) for c in CONDITIONS]
    # Added later, at the END so every earlier job keeps its worker: the look-alike pairs
    # (Step L1, 5 runs) and the sealed test set (Step 6, 10 runs: they are the headline).
    if part in (None, "pairs"):
        planned += [(s, c, "en", run) for run in range(1, RUNS + 1)
                    for s in PAIR_SCENARIOS for c in CONDITIONS]
    if part in (None, "heldout"):
        planned += [(s, c, "en", run) for run in range(1, HELDOUT_RUNS + 1)
                    for s in HELDOUT_SCENARIOS for c in CONDITIONS]
    return planned


def key(scenario_name, condition, language, run):
    return (scenario_name, condition, language, run)


def final_records():
    """{key: record} for every final run that counts: this batch and agent model, a scenario
    that still exists with the same task wording in its language, the same verifier and judge
    (for runs with UNDO), and no API error (those are redone). The latest copy of a run wins."""
    tasks = {(s["name"], lang): in_language(s, lang)["task"]
             for s in all_scenarios() for lang in ["en"] + ARABIC if translated(s["name"], lang)}
    tasks.update({(s["name"], "en"): s["task"] for s in PAIR_SCENARIOS + HELDOUT_SCENARIOS})
    latest = {}
    if FINAL_FILE.exists():
        for line in FINAL_FILE.open(encoding="utf-8"):
            r = json.loads(line)
            language = r.get("language", "en")
            if (r.get("batch") == BATCH and r.get("model") == config.AGENT_MODEL
                    and tasks.get((r["scenario"], language)) == r.get("task")
                    and (not settings(r["condition"])["undo"]    # no UNDO: no verifier to match
                         or (r.get("verifier") == VERIFIER
                             and r.get("judge_model") == config.JUDGE_MODEL))
                    and r.get("stop_reason") != "error" and not r.get("judge_error")):
                latest[key(r["scenario"], r["condition"], language, r["run"])] = r
    return latest


def save_shared(record):
    """Append one run while other workers may be appending too. A lock file lets only one
    process write at a time, so two lines can never be mixed together. A lock left behind by a
    crashed worker is removed after 60 seconds."""
    lock = FINAL_FILE.with_suffix(".lock")
    while True:
        try:
            handle = os.open(lock, os.O_CREAT | os.O_EXCL)
            break
        except (FileExistsError, PermissionError):   # taken (or, on Windows, being deleted)
            try:
                if time.time() - lock.stat().st_mtime > 60:
                    lock.unlink()
            except OSError:              # gone already, or still open in another program
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
    parser.add_argument("--part", choices=["english", "arabic", "pairs", "heldout"],
                        help="run only one part")
    parser.add_argument("--worker", default="1/1", metavar="I/N",
                        help="run share I of N side by side (each gets 30/N requests a minute)")
    args = parser.parse_args()
    import freeze
    changed = freeze.differences()
    if changed and not args.estimate:
        raise SystemExit("Settings differ from FROZEN.md, so no final run may start:\n"
                         + "\n".join(f"  {k}: frozen {a!r}, now {b!r}" for k, a, b in changed)
                         + "\nRun `python freeze.py` (and note why in FROZEN.md) first.")
    i, n = map(int, args.worker.split("/"))
    # All workers share one throttle (llm.SHARED_CLOCK): together at most 30 requests a minute,
    # and a worker that is waiting for the model leaves its turns to the others.
    llm.SHARED_CLOCK = FINAL_FILE.parent / "llm_clock.txt"

    done = final_records()
    mine = jobs(args.part)[i - 1::n]           # a fixed share, so workers never overlap
    todo = [j for j in mine if key(j[0]["name"], j[1], j[2], j[3]) not in done]
    print(f"Final data: batch {BATCH!r}, agent {config.AGENT_MODEL}, judge {config.JUDGE_MODEL}, "
          f"verifier {VERIFIER}, {RUNS} runs each.")
    print(f"Worker {i}/{n}: {len(mine)} of {len(jobs(args.part))} runs are this worker's, "
          f"{len(mine) - len(todo)} saved, {len(todo)} to do "
          f"(all workers share {llm.MAX_REQUESTS_PER_MINUTE} requests/min).")
    rate = llm.MAX_REQUESTS_PER_MINUTE / n
    print(f"API calls: about {len(todo) * TYPICAL_CALLS_PER_RUN} typical "
          f"(~{len(todo) * TYPICAL_CALLS_PER_RUN / rate / 60:.1f} h at {rate:.0f}/min), "
          f"at most {len(todo) * MOST_CALLS_PER_RUN} "
          f"(~{len(todo) * MOST_CALLS_PER_RUN / rate / 60:.1f} h). A run also waits ~1 min "
          f"for the model, so expect about {len(todo) / 60:.0f} h for this worker.")
    if args.estimate:
        return

    for pass_number in range(1, MAX_PASSES + 1):
        if pass_number > 1:
            todo = [j for j in mine if key(j[0]["name"], j[1], j[2], j[3]) not in final_records()]
            if not todo:
                break
            print(f"\nPass {pass_number}: retrying {len(todo)} run(s) that failed.", flush=True)
        fails_in_a_row = 0
        for i, (scenario, condition, language, run) in enumerate(todo, 1):
            print(f"[{i}/{len(todo)}] {scenario['name']} [{condition}] {language} run {run}", flush=True)
            # The report (Step 5's honesty check) is asked only where it is analysed: no UNDO and
            # full UNDO. It costs a few API calls per run.
            record = run_one(scenario, condition, run_number=run, batch=BATCH, verifier=VERIFIER,
                             language=language, ask_report=condition in CONDITIONS)
            save_shared(record)       # saved even when it failed: we append, never delete
            print(f"      outcome={record['outcome']} decision={record['decision']} "
                  f"rolled_back={record['rolled_back']} stop={record['stop_reason']}", flush=True)
            failed = record["stop_reason"] == "error" or record["judge_error"]
            fails_in_a_row = fails_in_a_row + 1 if failed else 0
            if fails_in_a_row >= FAILS_BEFORE_PAUSE:
                print(f"      {fails_in_a_row} failures in a row: pausing {PAUSE_SECONDS}s", flush=True)
                time.sleep(PAUSE_SECONDS)
                fails_in_a_row = 0


if __name__ == "__main__":
    main()
