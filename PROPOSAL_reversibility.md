# Per-call reversibility for MCP tools: a discussion draft

*For MCP's Tool Annotations Interest Group. A draft for discussion, not a formal SEP. It has
not been posted anywhere.*

## 1. The problem, with numbers

AI agents now act through tools, and some of their mistakes are hard or impossible to take
back: a production database deleted during a code freeze, a file overwritten by a careless
rename, a refund promised against policy, customer data emailed to an attacker after a prompt
injection. Today's main defence is to **ask permission**, but people approve almost everything
they are asked (Anthropic reported a 93% approval rate for its coding assistant's permission prompts in
March 2026), so asking more often does not make agents much safer.

We built **UNDO**, a runtime that stages every tool call so it can be undone, holds or gates
the irreversible ones, and lets a verifier keep or roll back the work at the end. On UndoBench
(tasks built from real incidents and prompt-injection attacks, a 120B agent model, October 2026):

- on a **sealed test set** of 12 tasks written after every setting was frozen, damage left at
  the end fell from **39.2% (47/120 runs) without UNDO to 9.2% (11/120) with it** (Fisher's
  exact test p = 6e-8); on the 38 development tasks, from 17.4% to 1.6%;
- a person was asked **0.20** times per task instead of 1.21 (83% fewer approvals); with the
  same simulated person approving 93% of requests, permission-only left 21.1% damage while
  asking 1.27 times per task, UNDO 12.6% while asking 0.21 times;
- turning off the gate or rollback made damage significantly worse.

## 2. What exists today, and what it misses

`ToolAnnotations` has four boolean hints: `readOnlyHint`, `destructiveHint`, `idempotentHint`
and `openWorldHint`. None says whether a change can be **taken back**, or how.

- **SEP-1984** (Comprehensive Tool Annotations) proposed six hints, including
  `reversibleHint`: "operations can be undone". In its discussion, one reviewer suggested
  `idempotentHint` and retries instead, and another argued that `reversibleHint` is the wrong
  shape, because reversibility is a property of a specific **call**, not a tool, proposing
  tiers (snapshot, compensable, not reversible) plus "unknown". The PR was closed by its author
  to sync with the Interest Group.
- **SEP-3172** proposed optional **recovery metadata** in `ToolAnnotations`: a tool could
  advertise a compensating operation for workflow recovery. It was closed in September 2026
  because new SEPs should first be developed with a Working Group.

What both miss, in our data:

1. **Reversibility belongs to the call.** In UNDO the same `send_email` is compensable when it
   tells a colleague a typo is fixed and **irreversible** when it carries an API key or another
   customer's data; `transfer_money` is compensable between a company's own accounts and
   irreversible to a stranger. In a prompt-injection test, an agent tricked into emailing
   customer data to an outside address **sent it in 3 of 3 runs** when every call got its
   tool's default kind ("email: compensable"), and was **stopped before sending every time it
   tried** when each call was classified on its own. In the final data, 12 of 37 emails got a
   different kind than the tool's default.
2. **A claim of reversibility is only useful if it is true.** A hint cannot be trusted from an
   untrusted server; a tested undo can.
3. **Time can make an action reversible.** Holding an email or a payment for a short window
   (like "Undo send") makes it fully cancellable until it leaves.

## 3. A design to discuss

Four small pieces, in order of how much they ask of the protocol:

1. **Per-call reversibility in tool results (or a dry-run response).** Instead of (or besides)
   a static hint, a server reports, for each call, one of `reversible` (restorable from saved
   state), `compensable` (a correcting action exists, e.g. a correction email),
   `irreversible`, or `unknown`, with a short reason. A client can then ask a person only about
   the calls that need it. For tools whose effect can't be known in advance, a **dry run**
   (UNDO runs raw SQL and shell commands on a copy of the world first and classifies them by
   what changed) is one way to produce it.
2. **A reference to the undo.** A reversible or compensable call names the tool and arguments
   that undo or compensate it (close to SEP-3172's recovery metadata, but per call), so an
   orchestrator can roll back a failed workflow.
3. **A hold window.** A server may hold outgoing effects for a declared number of seconds and
   expose `cancel`/`approve` for held effects, so "irreversible" becomes "cancellable for N
   seconds".
4. **A conformance test.** A tool that claims to be reversible ships a test: snapshot the
   world, call, undo, and compare (UNDO's `undo_conformance.py` runs each action 100 times with
   random inputs; it caught a real bug in our own undo on its first run, and a deliberately
   broken undo in 95 of 100 runs).

## 4. The five questions (from the IG's March 2026 post)

1. **What client behaviour does it enable?** Ask a person only about irreversible calls (83%
   fewer approvals in our data); roll back reversible ones automatically when a check fails;
   offer "cancel" during a hold window; show compensable effects as "can be corrected".
2. **Does it need trust to be useful?** A *claim* needs trust, like every hint. The design is
   meant to still help when servers are wrong: `unknown` is the default and treated as
   irreversible; a conformance test (piece 4) and client-side dry runs (piece 1) let a client
   **check** instead of trust.
3. **Could `_meta` handle it?** For prototyping, yes: we would start with a namespaced `_meta`
   field. But clients only act on what they recognize, and "ask a person or not" is exactly the
   kind of behaviour that needs a shared vocabulary.
4. **Does it help reason about combinations?** Partly. Per-call kinds catch the
   *exfiltration* leg of the "lethal trifecta" at the moment it happens (an email carrying
   private data to an outside address becomes irreversible), and a session-level rollback
   undoes a chain of calls in reverse order. It does not, by itself, describe which tools are
   dangerous together.
5. **Is it a hint or a contract?** The kind is a hint; the conformance test makes it closer to
   a contract for servers that ship one. Enforcement (holding, gating, rolling back) stays in
   the client or runtime, as the post recommends.

## 5. Open questions and limitations

- Our evidence comes from **one fake company**, 38 tasks, a few agent models and one judge
  model. Real systems are bigger, and real undo depends on timing and on other services.
- **Classifying a call can need context** (is this address outside the company? does this
  email carry a secret?). Who provides that context, the server or the client?
- **Dry runs cost something:** copying the world grew in proportion to its size in our
  measurements; real systems would need copy-on-write or narrower previews.
- **Compensation is not undo.** A correction email does not un-read an email; our model treats
  a correction sent after the email was read as not undoing it.
- **A verifier that reads agent-written text can be misled.** With a plausible excuse in the
  agent's note, our AI judge still accepted 1 of 6 subtle harms.

We would be glad to share the code, the data and the conformance tests, and to adapt the
design to the Interest Group's direction.
