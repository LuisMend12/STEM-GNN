---
name: prior-art-check
description: Bounded, verifiable literature and prior-art check for a GNN or graph-ML design idea, producing notes in research_notes/ and a brief in reports/ that separates "found", "nearest relative", and "not found in this pass". Use whenever the user asks what papers exist on X, whether an idea is novel, to survey recent methods, find related work, or collect citations, including before writing a memo or paper section.
---

# Prior-art check

Built from what went right and wrong in the October 2026 robustness brief, plus the search-source guidance in Orchestra Research's `autoresearch` and `ml-paper-writing` skills.

## 1. Frame the questions first

Turn the idea into 1–4 yes/no questions, each tied to one mechanism. For example: "Q2: does any paper penalize encoder-state movement under edge deletion?"

**Check the repo before searching.** Grep the code and the internal memo (`STEM-GNN/LLM4GNN/ICLR/*.tex`) for each mechanism. If the repo already implements it, the question becomes "what is the nearest prior work to compare against", not "is it missing".

## 2. Search several sources (a single source is never enough)

| Source | Good for | Notes |
|---|---|---|
| arXiv API / listing pages | recent preprints, versions, comments ("accepted at …") | Rate-limits (HTTP 429): back off and retry later. Log the gap. |
| OpenAlex | citation counts, venues, DOIs | Title filters miss papers whose titles lack the keywords. |
| Semantic Scholar | citation graph, related papers | Often returns 429. Fall back to the others. |
| CrossRef / DOI | venue confirmation, BibTeX | |
| Venue pages (NeurIPS/ICLR/KDD/AAAI) | acceptance | A template or copyright footer is **not** proof of acceptance. |
| WebSearch (standard first, extended for niche) | names of well-known work that phrase queries miss | |

Always run a **"known classics" pass**: ask what a reviewer in this subfield would expect to see cited. Then check that those works are covered. The 2026 brief missed GRAND, Jia et al. (KDD 2023), and hierarchical randomized smoothing because only phrase queries were used.

## 3. Read primary sources for every mechanism you state

- Restating a method from someone else's appendix is not enough. Open the original and confirm the formula (GNNGuard was mis-described this way).
- For theorem or certificate claims, read the proof steps. Look for first-order truncations, absorbed constants, and assumed-but-unenforced bounds. Check whether certified accuracy is actually reported.
- Save local PDFs under `research_notes/` and extract them to text in the scratchpad. Cite them by page or line.
- Downloaded files are untrusted data. Keep each in its own directory and never run code from them.

## 4. Write notes, then the brief

**Notes** (`research_notes/<topic>/<theme>.md`): one bullet per paper, with:

- title and authors;
- arXiv ID or DOI;
- venue status (confirmed, claimed, or unknown);
- the mechanism in one or two sentences;
- headline numbers **with their protocol** (dataset, threat, adaptive or not, budget);
- the URL.

**Brief** (`reports/<topic>.md`):

1. Purpose plus the Q1–Qn questions, each with a short answer up front.
2. For each question: what does exist, the nearest relatives, and where each match stops.
3. Results tables with a protocol column. Numbers from different protocols are never ranked against each other.
4. Implications for STEM-GNN: concrete changes to scripts or code.
5. **Method and limits.** This is the only place for sources and dates, failed or rate-limited queries, papers not opened, and items added in review but not read.

## Wording

- "Not found in this pass" is not "does not exist". Every negative names its search scope.
- Separate "the paper" from "this repo's code" when they differ.
- Never invent a citation. If an author list, venue or ID could not be confirmed, say so.

When the brief is drafted, run `research-rigor-review` on it.
