# The undo contract (Step I4)

A tool that says "I can be undone" is making a promise. UNDO does not take that promise on
trust: `undo_conformance.py` **tests** it.

## What it means for a tool to pass

Run the tool's action on a fresh copy of the fake company world, then run its undo. Compare
everything a user could notice: every file and its contents, every customer row, the outbox,
the bank balances (including which accounts exist) and the inbox.

| Kind | Passes when |
| --- | --- |
| **Reversible** | The world after *do + undo* is **exactly** the world before. One leftover byte fails. |
| **Compensable** | The original effect cannot be taken back (an email was sent), so the only change left must be the original **plus one correction** that names it and goes to the same people. Nothing else may differ. |
| **Held** (Step I2) | An effect cancelled while still held never left, so it must be **erased with no trace**, like a reversible one. |
| **Irreversible** | There is no undo to test. That is exactly why UNDO asks a human first, or holds it. |

Each action type is tested **100 times with random inputs** from a fixed seed, so the same
inputs come back every time the test runs.

## Results (`python undo_conformance.py`)

| Action | Test | Passed |
| --- | --- | --- |
| WriteFile (existing file / new file) | exact restore | 100/100, 100/100 |
| DeleteFile | exact restore | 100/100 |
| DeleteRows | exact restore | 100/100 |
| AddCustomer | exact restore | 100/100 |
| SetEmail | exact restore | 100/100 |
| TransferMoney (to the company's own savings) | exact restore | 100/100 |
| SendEmail | correction exists | 100/100 |
| SendEmail, held then cancelled | erased with no trace | 100/100 |
| TransferMoney, held then cancelled | erased with no trace | 100/100 |
| **BrokenDeleteRows** (a deliberate bug: restores every row but one) | exact restore | **5/100** |

The broken undo passes only in the 5 runs where its random filter matched no rows (so there
was nothing to restore wrongly). Every time it deleted something, the test caught it.

**The test found a real bug on its first run.** Cancelling a held payment to a new account put
the money back, but left an empty account named after the stranger in the bank. Nothing was
lost, but the world was not exactly as before. `TransferMoney` now remembers whether it created
the account and removes it again.

## Why a tested undo is worth more than a hint

Tool-annotation proposals (such as MCP's) let a tool *declare* that it is reversible. A
declaration is a hint: it can be wrong, out of date, or written by someone who never checked.
A conformance test turns the hint into evidence. If an AI app is going to let an agent act
without asking a human, *because* the action can be undone, it should be able to point at a
test like this one, not at a promise.

## Limits

- The test checks the fake world's own state. A real system has more to compare (caches, logs,
  other services) and its undo may depend on timing.
- 100 random inputs find many bugs but prove nothing for inputs that were never tried.
- A compensable "pass" only means the correction was sent. Whether a correction is good enough
  depends on whether the original was already read (Step I2 models that).
