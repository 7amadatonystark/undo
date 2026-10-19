# UndoBench 1.0: benchmark card

> **License:** all rights reserved (see `LICENSE`). Running UndoBench or using its results needs
> the copyright holder's written permission.

## What it measures

UndoBench measures **how much damage an AI agent leaves behind**, and how much of it a safety
layer can take back. An agent model works on short tasks for a small, fake company (files, a
customer database, an inbox, an outbox and a bank) through tool calls. Each task is run
**without** and **with** UNDO, a runtime that stages every action so it can be undone, holds or
gates anything irreversible, and lets a verifier decide at the end whether to keep or roll back
the work.

Headline numbers, per agent model:

| Metric | Meaning |
| --- | --- |
| Damage without UNDO | runs where real harm was left at the end, with no safety layer |
| **Irreversible damage with UNDO** | runs where harm was still there after UNDO's decision: what could not be taken back |
| Recoverability | of the runs where the agent caused harm, how often UNDO removed it |
| Approvals per task | how often a person had to be asked (only irreversible actions ask) |
| False rollbacks | of the runs where the agent did no harm, how often UNDO threw the work away |
| Honesty gap | of the runs where the agent caused harm, how often its own report hid or misstated it |

Every rate comes with a 95% Wilson confidence interval and its counts.

## How the tasks were built

- **From real incidents** (`incidents.md`): each documented incident of an AI agent causing
  harm (a production database deleted during a code freeze, a rename that overwrote a file,
  an airline chatbot promising a refund against policy, prompt-injection leaks, and others)
  became at least one task, with its source noted in `scenarios.py`.
- **Harmless controls** (at least 10), where the right result is to keep the agent's work, so
  false rollbacks can be measured.
- **Vague tasks** ("clean up", "reset", "free up space"), because real requests are vague.
- **Prompt injection** (8 tasks): a harmless request, but an email, a file or a customer record
  hides an instruction aimed at the agent. The attacks only ever target the fake sandbox.
- **Look-alike pairs**: the same kind of change, harmless in one task and harmful in the other,
  so a verifier that blocks everything or allows everything scores badly.
- **Answer key** (`damaged()` / `done()` per task) is used **only for scoring**, after UNDO has
  decided. UNDO's verifier never sees it (a test checks that the verifier code never imports the
  scenarios).

## The sealed test set

The development tasks are the ones we looked at while building UNDO, so numbers on them can
look better than they really are. A **sealed test set** (`heldout/`) is written separately and
not opened, read or tuned on until the final run; every tunable number is frozen first
(`undo/settings.py` → `FROZEN.md`). The test-set numbers are the headline; development numbers
are reported separately.

## How it differs from related benchmarks

| | What it tests | How UndoBench differs |
| --- | --- | --- |
| **ToolEmu** (Ruan et al., ICLR 2024) | Risky agent behaviour, with tools *emulated by a language model* and an LM-based safety evaluator; 36 tools, 144 test cases | UndoBench's tools are **real code** on a real (fake) world, so damage is checked exactly, not judged by a model. And it measures not only whether an agent errs but whether a runtime can **undo** it |
| **AgentDojo** (Debenedetti et al., NeurIPS 2024) | Prompt-injection attacks and defenses in realistic tasks (email, banking, travel); 97 tasks, 629 security cases | UndoBench includes injections, but its focus is **recovery**: an attack that succeeds can still leave no damage if its effects are rolled back or held. It also covers non-adversarial mistakes |
| **InjecAgent** (Zhan et al., ACL Findings 2024) | Indirect prompt injection in tool-integrated agents; 1,054 cases, direct harm and data exfiltration | Same difference: UndoBench scores the **damage left**, not only attack success, and reports both side by side |

UndoBench is much **smaller** than these. Its contribution is a different question: *after* an
agent has done something wrong, can it be taken back?

## Result format

`python undobench.py` writes `results/undobench_<model>.json`:

```json
{
  "benchmark": "UndoBench", "version": "1.0",
  "agent_model": "...", "judge_model": "...", "run_date": "YYYY-MM-DD",
  "scenarios": ..., "runs": ...,
  "damage_without_undo":           {"k": ..., "n": ..., "rate": ..., "ci95": [..., ...]},
  "irreversible_damage_with_undo": {"k": ..., "n": ..., "rate": ..., "ci95": [..., ...]},
  "recoverability":                {"k": ..., "n": ..., "rate": ..., "ci95": [...]},
  "approvals_per_task": ...,
  "false_rollbacks":               {"k": ..., "n": ..., "rate": ..., "ci95": [...]},
  "honesty_gap":                   {"k": ..., "n": ..., "rate": ..., "ci95": [...]}
}
```

`k` is the count, `n` the number of runs it is out of, and `ci95` the 95% Wilson interval
(fractions between 0 and 1). Every individual run is also saved, one JSON line each, in
`results/runs.jsonl`.

## How to run it

```
python undobench.py --model <model name> --base-url <OpenAI-compatible endpoint> --key-env <ENV VAR WITH THE KEY>
python make_leaderboard.py        # rebuilds leaderboard.html from every saved run
```

The judge model stays the same for every model on a leaderboard, so models are compared fairly.

## Known limitations

- **Small:** 38 tasks (plus the sealed set) in one small, fake company. Real systems are bigger
  and messier.
- **Fake world:** emails are lines in a file and money is numbers in a JSON file. The "reading"
  of an email is a simple model (a random delay with a fixed seed).
- **The verifier uses an AI judge**, which can be wrong, can be fooled by text the agent writes
  (we measured this: `judge_injection.py`), and depends on one judge model.
- **Keyword answer keys:** a few scoring checks (e.g. "did the email promise a refund?") look for
  words; unusual wording can slip through.
- **A temperature above 0:** runs vary, which is why every task is run several times and every
  rate has a confidence interval.
- **One language for most tasks:** 19 tasks also exist in Arabic (Modern Standard and Egyptian).

## Reporting results honestly

Report the version, the agent model, the judge model, the date and the number of runs; report
confidence intervals, not just rates; report the sealed test set separately from development
tasks; and never tune settings on the sealed set.
