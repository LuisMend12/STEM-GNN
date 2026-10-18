---
name: research-rigor-review
description: Independent, evidence-checked review of a research artifact in this repo (literature brief, experiment analysis, findings, memo or paper draft) using two reviewers from different model families (a Claude subagent and Codex via the codex plugin), then a verified merge and revision. Use whenever the user asks to review, revise, check, critique, fact-check, or "make rigorous" a report, brief, notes, results, or paper section, or before any robustness claim leaves the repo.
---

# Research rigor review (dual-model)

This skill turns the review process used on `reports/Recent GNN robustness methods.md` into a repeatable procedure. Its six scoring dimensions are adapted from Orchestra Research's `ara-rigor-reviewer` and work on ordinary markdown and LaTeX instead of ARA directories.

## 1. Scope the review

Collect the following:

- the target file;
- its sources: notes, PDFs, results json, and the code the artifact talks about;
- the question it is supposed to answer;
- today's date, needed for venue and arXiv-ID plausibility checks.

## 2. Launch two independent reviewers in parallel

Run both in the background and give them the same task text:

- `Agent(subagent_type="general-purpose")`: the Claude reviewer. It may use the web.
- `Agent(subagent_type="codex:codex-rescue")`: hands the task to Codex.

Rules for both:

- They are **read-only** on the repo.
- Each writes its review to its own file in the scratchpad (`review_claude.md`, `review_codex.md`).
- Neither sees the other's output.

Each reviewer scores D1–D6 (1–5) and lists findings. Every finding needs:

- a severity (HIGH, MEDIUM or LOW);
- the **exact quoted text**;
- the problem;
- the evidence (`file:line`, a PDF page or line, or a URL);
- a proposed replacement.

Each review ends with a prioritized top 10.

| Dim | Check |
|---|---|
| D1 Evidence relevance | Does the cited source actually support the sentence? Is the claim type matched to the evidence: causal needs ablation, "robust" needs adaptive attack, "better" needs a baseline? |
| D2 Falsifiability | Is there a stated condition that would refute each claim? |
| D3 Scope calibration | Look for over-claims ("not found" read as "does not exist", one dataset read as general) and under-claims (findings in the notes that the artifact omits). |
| D4 Coherence | Do the intro, body, tables and conclusion answer the same questions? |
| D5 Exploration integrity | Are negative results and failed searches recorded honestly and in one place? |
| D6 Methodological rigor | Check baselines, variance, protocol labels, and code-versus-paper drift (the repo may already implement what the text calls missing). |

Also ask both reviewers for **relevance to STEM-GNN**: concrete implications for the scripts and code.

## 3. Verify before merging

Reviewer output is model output, not ground truth. For every HIGH finding, and for each finding where the reviewers disagree, check the source yourself:

- grep the code;
- read the PDF text around the cited equation;
- re-read the note line.

Findings that **both** reviewers raised independently get top priority. Drop or soften anything you could not confirm. Mark items that a reviewer found on the web and that nobody read in full as "added in review, not read in full".

## 4. Revise

- Apply the confirmed findings directly to the artifact. Keep its sources and citations.
- Put the purpose and short answers first.
- Move scattered provenance caveats into one "Method and limits" section.
- Save both raw reviews next to the artifact (for example `reports/reviews/`) so the revision can be traced.
- Do not commit unless the user asks.

## 5. Report to the user

Give the user four things:

- what both reviewers agreed on;
- what came only from Codex and only from Claude;
- what you verified and how;
- anything left unverified.

Then ask before committing.

## Rules

- Quote verbatim. A finding without a locatable quote or source is a suggestion, not an error.
- Every weakness comes with a proposed fix.
- Calibrate: most competent drafts score 3–4 per dimension.
- If a reviewer reports being blocked by a policy or permission, surface that to the user. Do not redo the blocked action on its behalf.
