# Stage 1 analysis: LLM-guided routing in STEM-GNN

Status: Codex-reviewed (`../reviews/codex_llm_stem_analysis_review_2026-10-31.md`), with its fixes applied. Protocol: `protocol.md`, Amendment 1. Reproduce with `python STEM-GNN/scripts/llm_stem/analyze.py`.

## Conclusion

Stage 1 did not establish a Qwen-specific robustness advantage on the preregistered worst-node endpoint. No hypothesis met its support threshold. H60 and H61 remain inconclusive under the evidence-against rule, because their upper limits (+1.08 and +1.31) are above +1, with point estimates near zero.

## Data

All 15 cells (3 splits x 5 seeds) and 6 arms are present, 90 rows. Penalty coefficient 0.001, 200 pretraining epochs. All 930 attacked graphs passed validation; the largest edit counts were 527 at 10% and 1,055 at 20%, which equal the budgets.

Split 0 seed 1 (job 32535027_1) was cancelled at 13 of 60 attacks because per-attack time had risen from about 5 to 16 minutes, and was rerun from the start with the same command as job 32537560_1, excluding two busy nodes. Logs are in `logs/`. The script emits metrics and result files only after all attacks finish, so the cancelled run produced no accuracy output, and the decision was based on speed. This isn't automatically unbiased: runtime could depend on model state, and retraining replaces one non-deterministic realization with another. The rerun's attacks near the end took about 45 seconds on the new node, which points to the node, but doesn't prove it.

Each arm was evaluated on the same candidate suite: at each budget, five attacks on each of the six arms plus SAGE transfer, along with lower budgets and the clean graph. Worst-node takes each test node's worst graph, so different nodes can use different graphs. Worst-graph is the single worst graph. Own-attack is the intersection of clean correctness and all ten of an arm's own attacks across both budgets. SAGE transfer is a single graph at the stated budget.

Exposure is symmetric, but attack effectiveness isn't equal. Removing an arm's own attacks raises its worst-node accuracy by 1.70 (S-RF) to 2.56 (S0) points. So these comparisons are conditional on this finite, model-generated suite.

## Hypotheses (worst-node accuracy at 20%, points)

| ID | Comparison | Mean | 95% CI | p | Cells won | Split 0 / 1 / 2 | Clean change | Verdict |
|---|---|---|---|---|---|---|---|---|
| H60 (primary) | S-R − S-R0 | −0.15 | [−1.37, +1.08] | 0.80 | 7/15 | +1.18 / −0.00 / −1.62 | +0.85 [−0.07, +1.77] | Not supported, inconclusive by rule |
| H61 | S-R − S0 | +0.03 | [−1.26, +1.31] | 0.97 | 6/15 | +0.44 / +0.84 / −1.20 | +0.79 [−0.12, +1.69] | Not supported, inconclusive by rule |
| H62 | S-R − max(S-U, S-RF, S-PQ) | −1.04 | [−2.17, +0.09] | 0.07 | 5/15 | +0.12 / −1.97 / −1.26 | n/a | Not supported |

Holm-adjusted p: H61 0.97, H62 0.14. H62 compares S-R with the best observed trivial router in each cell, as preregistered, rather than a single fixed router, so its negative estimate doesn't establish inferiority to every trivial router. S-R minus each one: S-U −0.19 [−1.48, +1.10], S-RF +0.05 [−0.78, +0.89], S-PQ −0.51 [−1.56, +0.53]. H62 has no preregistered evidence-against threshold. Both clean-change conditions are met, but H60 and H61 fail on robustness.

## All arms (mean ± SD over cells, points)

| Arm | Router reads | Validation | Clean | Worst-node 10% | Worst-node 20% | Worst-graph 20% | Own-attack 20% | SAGE transfer 20% |
|---|---|---|---|---|---|---|---|---|
| S0 | Message-passed z | 79.17 | 77.30 ± 1.09 | 43.08 | 33.19 ± 2.68 | 55.36 | 45.87 | 65.16 |
| S-R0 | MiniLM text | 79.03 | 77.23 ± 1.64 | 42.87 | 33.36 ± 1.93 | 55.41 | 45.95 | 65.35 |
| S-R | Qwen embedding | 79.60 | 78.08 ± 1.53 | 42.92 | 33.22 ± 2.97 | 56.30 | 47.03 | 66.01 |
| S-U | Frozen uniform | 79.32 | 78.10 ± 1.80 | 43.13 | 33.41 ± 1.80 | 56.27 | 47.31 | 64.97 |
| S-RF | Fixed random features | 78.93 | 77.55 ± 1.51 | 42.75 | 33.16 ± 2.78 | 55.48 | 46.33 | 64.03 |
| S-PQ | Permuted Qwen | 79.05 | 77.58 ± 1.83 | 43.60 | 33.73 ± 2.29 | 55.71 | 45.94 | 65.01 |

## Descriptive results (no multiplicity control, not confirmatory)

- S-R − S0 worst-graph at 20%: +0.93 [+0.10, +1.77], p = 0.03, 12/15. S-U − S0 has a similar point estimate, +0.91 [−0.13, +1.94]. The paired S-R − S-U difference is +0.03 [−1.11, +1.16], so these descriptive results don't establish a Qwen-specific benefit.
- S-U − S0 own attack at 20%: +1.44 [+0.07, +2.82], p = 0.04. S-R − S0: +1.16 [−0.26, +2.58]. These are upper bounds on robust accuracy. They are consistent with a small benefit from removing edge-steered routing, but this is one test among dozens.
- S-RF − S0 SAGE transfer at 20%: −1.13 [−2.10, −0.17], p = 0.02. Also uncorrected.

## Diagnostics

| Arm | Routing shift, layers 1 / 2 | Expert perplexity, layers 1 / 2 (max 3) | Best epoch |
|---|---|---|---|
| S0 | 0.000 / 0.044 | 2.47 / 2.04 | 31.3 |
| S-R0 | 0.000 / 0.000 | 2.54 / 2.65 | 31.3 |
| S-R | 0.000 / 0.000 | 1.72 / 2.24 | 42.3 |
| S-U | 0.000 / 0.000 | 3.00 / 3.00 | 36.5 |
| S-RF | 0.000 / 0.000 | 2.97 / 2.97 | 48.7 |
| S-PQ | 0.000 / 0.000 | 2.09 / 2.44 | 33.3 |

Routing shift is mean total variation over test nodes, per layer, on each arm's own `ste` attack at 20%. S0's layer-2 value of 0.044 is an average on that one attack, and this diagnostic alone doesn't explain the small worst-node effect. S-R's zero shift is an implementation check, not evidence of cause. Expert perplexity is computed from clean, all-node mean soft routing probabilities, per layer. Code change rates are per VQ head on test nodes under the same attack, about 0.26–0.27 in all arms.

**Preregistered but not reported:** expert-output shift (needed to assess the proposed mechanism) and codebook perplexity weren't collected. Expert perplexity is the only expert-usage measure. Teacher-error diagnostics don't apply, because Stage 1 doesn't use the Qwen prior.

## Limits

One graph (Cora), with intervals conditional on it; three splits are not independent replicates, and split effects vary in sign. Single LLM and embedding. Depth 2 only, where routing has few chances to matter. The attacker uses test labels. The attack suite is finite, so these are upper bounds on robust accuracy, not certificates. Non-deterministic reruns are expected. S0 uses the EMA repair and X2 recipe, not the full STEM objective. Stage 1 includes no plain non-STEM baseline in the robust comparison and no MIX or MLP controls. Text attacks are out of scope, unlabeled nodes' text is used transductively, 21.8% of texts are truncated at 1,200 characters, Qwen may have seen Cora in its own pretraining, and aggregate Qwen test accuracy had been seen before this study.

## Consequence for Stage 2

The protocol makes Stage 2 conditional on Stage 1's analysis and review, not on a positive result. The router mechanism gives no reason to include S-R as the main component. Other placements (input swap, semantic distillation at the VQ, anchor at the head) remain untested.
