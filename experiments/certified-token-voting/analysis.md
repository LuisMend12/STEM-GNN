# Analysis: Certified Token Voting (CTV), Cora, 3 splits

Status: **H11 not supported under a strong attack.** The PRBCD-based result was an attack-strength artifact. The certificate is sound (0 violations) but does not beat the trivial structure-free baseline.

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
| **H11** adaptive robustness | **Not supported** | PRBCD said 68.0 at 20%, but the exact-greedy attack drives the locked CTV to 46.5. PRBCD overestimated it by about 21 points: with soft votes the scores are nearly flat and PRBCD's search is weak. |
| **H12** certificate non-trivial | **Not supported** | The locked setting certifies 30% at k=1. Sharp-vote settings reach 73% at k=1, which does not beat the structure-free MLP (72.5%). As α grows, CTV converges to the MLP. |
| **H13** radius grows with degree | **Supported** | Mean radius of correct nodes: 0 for degree < 2, 0.3 for degree 2–3, 0.7 for degree 4–7, and 2.2 for degree ≥ 8 (split 0, locked). |
| **H14** certificate sound | **Supported** | 0 violations. |

## What we learned

1. **PRBCD can badly overestimate robustness for linear and voting models.** The locked CTV looked like the most robust model in the study (68.0 at 20%) until an exact attack was run (46.5). This is a methodological lesson: whenever a model's flip cost can be computed exactly, compute it. Read every PRBCD number in this repo, including SAGE's, as an optimistic upper bound.
2. **Sharper votes trade clean accuracy for real robustness.** α=4 with T=0.1 holds 59.0 under the exact attack, with a median flip cost of 6 edits, at 75.1 clean. That is the best *verified* robust accuracy in this study. However, α and T were picked from a frontier computed on split 0's *test* set, so it is exploratory. A confirmatory run must select α and T on validation, for example by maximizing validation exact-greedy accuracy at 10%.
3. **Structural certification on Cora mostly restates "trust the text".** The certificate's value is bounded above by what the text alone already gives. CTV becomes useful only if the votes are *better than the text-only classifier*, for example with a stronger LLM prior such as Qwen, or with multi-hop votes. Otherwise the best certified model is just the MLP.

## Next steps

- **Confirmatory α/T selection on validation** using the exact-greedy attack, then rerun on all 3 splits.
- **Qwen prior for the votes** (75% zero-shot on 16 nodes). That is the lever that could lift CTV's certified accuracy above the MLP line. It needs the GPU environment, which is pending the user's go-ahead.
- **Prior-art check** before any novelty claim: bounded-vote and label-propagation certificates, AGNNCert-style counting, and Bojchevski et al. (2020).
- **Exact or stronger attacks for SAGE**, such as PRBCD with more epochs and restarts, so the SAGE-vs-CTV comparison is fair.
