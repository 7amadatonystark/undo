# Human-participants study: draft for IRB review

**Status: DRAFT, not approved.** Nobody other than the researcher may use `study_app.py` until
the fair's Institutional Review Board (IRB) has given written approval. The app stays in
practice mode (`IRB_APPROVED` is not set) until then. Before submitting, fill in every
`[bracketed]` item and copy the text into the fair's official forms (for ISEF: the Research
Plan, Form 4 Human Participants and the risk assessment, plus the consent forms).

## 1. Title and researcher

- **Title:** Does asking about every AI action tire people out? Approval fatigue with and
  without an undo layer.
- **Researcher:** Abdullah Bedir, [school], grade [ ].
- **Adult sponsor / qualified scientist:** [name, contact].
- **Where:** [school / fair site], on the researcher's laptop. No internet service is used.

## 2. Research question and purpose

AI assistants increasingly ask people to approve their actions ("May I send this email?"). When
people are asked about everything, they may stop reading and approve automatically, including
harmful actions. This study tests whether people approve fewer harmful actions when they are
asked only about actions that cannot be undone (with the UNDO safety layer) than when they are
asked about every action.

**Hypothesis:** participants in condition B (asked only about irreversible actions) approve a
smaller share of the harmful actions than participants in condition A (asked about every action).

## 3. Participants

- **Who:** [number, e.g. 30-40] volunteers aged [range, e.g. 14-18 and adults], recruited from
  [school classes / clubs] by [an announcement / poster], with no reward [or: describe].
- **Excluded:** nobody is excluded except people who do not consent (or whose parent does not,
  for minors).
- **Vulnerable groups:** minors may take part, with parental permission and their own assent.
  No other vulnerable groups are targeted.

## 4. What participants do (about 10 minutes)

1. Read the information sheet and give consent (and parental permission for minors).
2. On the laptop, open the study page (`streamlit run study_app.py`). It assigns a random ID and
   one of two conditions, in turn, so both groups are the same size.
3. Watch a scripted (pre-written) AI assistant work through 20 short office tasks in a **fake**
   company. Whenever it asks, click **Approve** or **Block**.
   - Condition A: it asks about every action (about 41 decisions).
   - Condition B: it asks only about actions that cannot be undone (about 7 decisions).
   - Four harmful actions (for example, sending a fake customer list to an outside address) are
     mixed in; both groups are asked about all four.
4. A thank-you screen. Participants may stop at any time without giving a reason.

No real email, money, files or personal data are involved: everything is fake and on-screen.

## 5. Risks and how they are reduced

| Risk | Level | How it is reduced |
| --- | --- | --- |
| Boredom or mild fatigue from clicking | minimal | about 10 minutes; can stop at any time |
| Feeling bad about approving a "harmful" action | minimal | everything is fake; the debrief explains that most people approve such actions and why |
| Privacy | minimal | no name, email, age or any personal information is recorded |

The study involves **no more than minimal risk**: nothing beyond everyday computer use.

## 6. Data collected and how it is protected

- **Recorded per participant:** a random ID, the condition (A or B), the date, the number of
  decisions, the number of harmful actions shown and approved, and the mean and median seconds
  per decision. **Nothing else.** No names, emails, ages, IP addresses or recordings.
- **Stored:** in `study_results.csv` on the researcher's laptop only. It is never uploaded.
- **Kept:** until the end of the fair season [date], then deleted.
- Consent forms (which carry names) are kept on paper, separately from the data, and cannot be
  linked to a random ID.

## 7. Analysis

`study_summary.py` compares the two conditions: decisions per person, the share of harmful
actions approved, and time per decision. Fisher's exact test checks whether the difference in
harmful approvals could be luck (with few participants this is reported as a pilot).

## 8. Debrief (read or shown at the end)

"Thank you! The assistant and the company were fake. We were testing whether being asked about
everything makes people approve things without reading. Half of the participants were asked
about every action, half only about actions that cannot be undone. Please don't tell other
possible participants about the harmful actions, so the study stays fair."

## 9. Informed consent form (adults) / assent form (minors)

> **Study:** Approval fatigue with AI assistants. **Researcher:** Abdullah Bedir, [school].
>
> You are invited to take part in a science-fair study. You will watch a pretend AI assistant
> do office tasks for a fake company and click Approve or Block when it asks. It takes about
> 10 minutes. Nothing is real: no real emails, money or files.
>
> **Risks:** none beyond everyday computer use; you might find it a little boring.
> **Privacy:** we do not record your name or anything that identifies you; your answers are
> saved under a random number. **Voluntary:** you can stop at any time, and saying no will not
> affect you in any way. **Questions:** [researcher contact], [adult sponsor contact].
>
> I have read this and agree to take part.
> Name: ______________________  Signature: ______________________  Date: __________

## 10. Parental permission form (for participants under 18)

> My child, ______________________, is invited to take part in the study described above. I
> have read the description, including the risks (minimal), the privacy protections (no
> personal information recorded) and that participation is voluntary and can stop at any time.
>
> [ ] I give permission for my child to take part.  [ ] I do not give permission.
> Parent/guardian name: ______________________  Signature: ______________________
> Date: __________

## 11. Checklist before any participant

- [ ] IRB written approval received (date: ________), and the approval form kept with the project.
- [ ] Forms filled in: research plan, human-participants form, risk assessment, consent/assent and parental permission.
- [ ] `IRB_APPROVED=yes` set in `.env` only after approval.
- [ ] Practice data (`study_practice.csv`) deleted before the first real participant.
