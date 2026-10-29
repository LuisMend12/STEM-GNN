# Analysis: Certified Token Voting (CTV), Cora, 3 splits

Status, revised after the Codex analysis review (`../reviews/codex_analysis_review_2026-10-20.md`):

- **H11's registered PRBCD criterion is met** (68.0 at 20% vs 61.0 for DropEdgeCons, a +7.0 margin against the +5 required).
- **Robustness under a stronger attack: resolved, and it is much weaker.** A *realized* greedy attack (`exact_attack.py::realized_attack`) builds the attacked graph, updates both endpoints after each edit, re-plans every target against the current graph, asserts the edit count is within budget, and evaluates the model on the final graph. On the locked CTV it reaches **59.3 at 10% and 46.6 at 20%**, against PRBCD's 74.5 and 68.0. The independent-cost estimate agreed with the realized attack to within ±0.5 points on every split and config, so endpoint interference is negligible on Cora.
- **The certificate is sound** (0 violations).

## Results (mean ± std over OFA splits 0/1/2, accuracy %)

| Model | Clean | Camouflage 20% | PRBCD 10% | PRBCD 20% | Exact-greedy 10% | Exact-greedy 20% | Cert@1 | Cert@2 | Cert@3 |
|---|---|---|---|---|---|---|---|---|---|
| MLP (text only, structure-free) | 72.5 | 72.5 | 72.5 | 72.5 | 72.5 | 72.5 | 72.5 | 72.5 | 72.5 |
| SAGE | 81.3 | 77.3 | 64.3 | 55.1 | n/a | n/a | — | — | — |
| SAGE + TextGate (raw cosine) | 79.1 | 74.1 | 66.0 | 60.0 | n/a | n/a | — | — | — |
| SAGE + DropEdgeCons (earlier run) | 82.9 | — | 68.8 | 61.0 | n/a | n/a | — | — | — |
| **CTV, locked (val-tuned, soft votes)** | 78.6±1.9 | 75.4 | 74.5 | 68.0±3.6 | **59.2** | **46.5** | 30.4 | 10.3 | 3.8 |
| CTV + pseudo-labels | 77.8 | 74.6 | 72.4 | 64.7 | — | — | 49.9 | 20.2 | 7.1 |
| CTV, α=2, T=0.1 [EXPL] | 77.8 | 75.8 | 66.3 | 57.1 | 62.2 | 53.7 | 71.1 | 62.3 | 44.6 |
| CTV, α=4, T=0.1 [EXPL] | 75.1 | 74.5 | 69.2 | 64.2 | 65.1 | 59.0 | 73.2 | 70.8 | 66.5 |

**Columns:**

- **Exact-greedy** (`exact_attack.py`) computes each test node's minimum flip cost exactly, since CTV is linear in the edges, then spends the global budget on the cheapest nodes first. It is stronger than PRBCD on CTV, but it is not provably the optimal global attack.
- **Cert@k** is the fraction of test nodes that are correct and certified against any k incident edge edits.
- **Certificate violations: 0** across all PRBCD graphs, so the certificate code agrees with the math.

## Hypothesis outcomes

| Hypothesis | Result | Detail |
|---|---|---|
| **H10** clean ≥ 78% | **Supported (borderline)** | 78.6 ± 1.9 for the locked setting. |
| **H11** adaptive robustness | **Registered PRBCD criterion met; NOT robust under a stronger attack** | PRBCD: 68.0 at 20% (+7.0 over DropEdgeCons). The realized greedy attack, which builds graphs and stays within budget, gives 46.6. PRBCD overestimates CTV by about 21 points. SAGE's 55.1 is also PRBCD-measured, so CTV vs SAGE under the strong attack is untested. |
| **H12** certificate non-trivial | **Not supported** (original 60%/40% rule) | The locked setting certifies 30.4% at k=1 and 10.3% at k=2. The MLP-relative restatement was written after the certificate results, so it is a post-result interpretation. The exploratory α=4 setting certifies 73.2% at k=1, a small numerical improvement over the MLP's 72.5% with no confirmatory evidence. |
| **H13** radius grows with degree | **Supported** | Mean radius of correct nodes: 0 for degree < 2, 0.3 for degree 2–3, 0.7 for degree 4–7, and 2.2 for degree ≥ 8 (split 0, locked). |
| **H14** certificate sound | **Supported** | 0 violations. |

## What we learned

1. **PRBCD can badly overestimate robustness for linear and voting models. Verified with realized attacked graphs:** 68.0 under PRBCD versus 46.6 under a graph-constructing greedy attack at 20%. The locked CTV looked like the most robust model in the study (68.0 at 20%) until an exact attack was run (46.5). This is a methodological lesson: whenever a model's flip cost can be computed exactly, compute it. Read every PRBCD number in this repo, including SAGE's, as an optimistic upper bound.
2. **Sharper votes trade clean accuracy for real robustness.** α=4 with T=0.1 holds 58.7 under the realized attack, with a median flip cost of 6 edits, at 75.1 clean. That is the best *verified* robust accuracy in this study. However, α and T were picked from a frontier computed on split 0's *test* set, so it is exploratory. A confirmatory run must select α and T on validation, for example by maximizing validation exact-greedy accuracy at 10%.
3. **Structural certification on Cora mostly restates "trust the text".** The text-only MLP is a useful comparison baseline, but *not* a mathematical upper bound, because neighbors can correct text-only errors. CTV becomes useful only if the votes are *better than the text-only classifier*, for example with a stronger LLM prior such as Qwen, or with multi-hop votes. Otherwise the best certified model is just the MLP.

## Next steps

- **Confirmatory α/T selection on validation** using the exact-greedy attack, then rerun on all 3 splits.
- **Qwen prior for the votes** (75% zero-shot on 16 nodes). That is the lever that could lift CTV's certified accuracy above the MLP line. It needs the GPU environment, which is pending the user's go-ahead.
- **Prior-art check** before any novelty claim: bounded-vote and label-propagation certificates, AGNNCert-style counting, and Bojchevski et al. (2020).
- **Exact or stronger attacks for SAGE**, such as PRBCD with more epochs and restarts, so the SAGE-vs-CTV comparison is fair.

## Realized-attack results (added after the Codex analysis review)

Each cell is per-split accuracy under the realized greedy attack, with the number of unique undirected edits used in parentheses. The 10% budget is 527 edits and the 20% budget is 1055.

| Config | Split | Realized 10% | Realized 20% |
|---|---|---|---|
| locked | 0 / 1 / 2 | 60.7 (527) / 61.4 (527) / 55.9 (527) | 48.3 (1055) / 48.8 (1055) / 42.6 (1055) |
| α=2, T=0.1 [EXPL] | 0 / 1 / 2 | 62.8 (525) / 64.3 (525) / 59.7 (527) | 54.0 (1051) / 55.4 (1053) / 51.0 (1055) |
| α=4, T=0.1 [EXPL] | 0 / 1 / 2 | 65.8 (527) / 67.2 (527) / 62.3 (527) | 59.5 (1055) / 60.9 (1055) / 55.7 (1055) |

Codex's two-node counterexample, run through `realized_attack`: the independent estimate says 0% accuracy, while the realized accuracy is 50%. This matches the review.
