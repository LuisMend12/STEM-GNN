# Protocol: capped graph correction against edge attacks

Written 2026-11-04, before any run. Code: `STEM-GNN/scripts/capped/run.py`.

## Question

Can STEM-GNN keep some of its clean-accuracy gain from the graph while limiting how much an edge attacker can change its predictions? In Stage 2 (`../llm-stem/analysis_stage2.md`), S0 scored 78.3 clean but 33.1 worst-node at 20%. The text-only MLP-M scored 70.0 under both, because edges can't reach it.

## Method

CAP(B): logits = m(x) + B · tanh(g(x, A) / B).

- m is MLP-M: the Stage 2 graph-free MLP on MiniLM features, trained first and then frozen.
- g is a STEM-GNN model (S0's pretrained MoE encoder, frozen VQ and linear head) finetuned *through* the combined output, so it learns a correction to m.
- Every correction coordinate lies in (−B, B) for any graph. Edges enter only through g. Text never weights, selects, adds or drops edges.

**Certificate (independent of the attack).** If m's top-class logit exceeds every other class's by more than 2B, the prediction equals m's for every graph. Certified accuracy is the fraction of test nodes where m is correct and its margin is above 2B. It is a lower bound on accuracy under any edge attack of any budget. It doesn't depend on PRBCD's strength, which matters because PRBCD has been optimistic in this repo.

**Cap grid, scale-free.** B_q = ½ × the q-quantile of m's validation-node margin, for q ∈ {0.1, 0.25, 0.5, 0.75, 0.9}. At B_q, roughly a fraction 1 − q of validation nodes meet the margin condition; certification also requires m to be correct. A nonpositive or nonfinite cap is rejected and the cell fails, rather than being clamped.

**Selection (CAP\*), using validation only.** For each cell, B\* maximizes the mean of clean validation accuracy and certified validation accuracy, with ties going to the smaller B. All five CAP(B) models are attacked. CAP\* is the confirmatory arm; the other values are descriptive and trace the trade-off.

## Arms (all attacked except MLP-M)

| Arm | Definition |
|---|---|
| S0 | Stage 2 S0: same pretraining, finetuning and frozen penalty coefficient 0.001 |
| RES | m(x) + g(x, A) with no cap, trained the same way. Tests whether the cap or just the MLP-plus-residual design matters |
| CAP(B_q) | The five capped models |
| MLP-M | Graph-free, not attacked; worst-node equals clean |

The attack suite and evaluation are unchanged from Stage 2:
- Five own PRBCD attacks per attacked arm at 10% and 20%, plus SAGE-surrogate transfer.
- Worst-node over the shared union of every graph built in the cell, plus clean.
- The attacker uses test labels.
- **BPDA attacks (added after the Codex design review, before any run).** Every CAP arm also gets two BPDA attacks, `bpda` (prob-margin loss) and `bpda_tanh` (tanh-margin loss), at both budgets. They use the exact capped forward with an identity backward through tanh and STE through the VQ, so tanh saturation can't hide gradients. Their graphs join the shared union.
- 15 cells: splits 0–2 × seeds 0–4, base seed 1000·seed + split, matching Stage 2 so S0 and MLP-M are reproduced.

## Hypotheses

n = 15 cells, two-sided paired t against a null of 0, 95% CIs, in percentage points, conditional on Cora. H80–H82 are one Holm family. Thresholds are on the point estimate, together with Holm p < .05; they aren't claims that the true effect exceeds the threshold.

| ID | Comparison | Endpoint | Support | Evidence against |
|---|---|---|---|---|
| H80 | CAP\* − MLP-M | Worst-node@20% | ≥ +1, Holm p < .05 | Upper CI < 0 |
| H81 | CAP\* − MLP-M | Clean | ≥ +2, Holm p < .05 | Upper CI < +1 |
| H82 | CAP\* − RES | Worst-node@20% | ≥ +2, Holm p < .05 | Upper CI < +1 |

**Claim rule.** "Capped correction improves edge robustness over STEM while keeping a graph benefit" requires H81 and H82, plus CAP\* − S0 worst-node@20% ≥ +2 with the CI excluding 0. If H80 also holds, CAP\* beats the graph-free model under attack. If H80 fails while H81 holds, the result is a trade-off rather than domination, and the report must say so. The CAP\* − S0 gate is a conjunctive claim condition (two-sided paired t, 95% CI), outside the Holm family. "Some graph benefit" means only H81; retained gain (CAP\* − MLP-M) / (S0 − MLP-M) on clean accuracy is reported descriptively, with no threshold.

**Descriptive:**
- CAP\* − S0 clean.
- Certified test accuracy per arm.
- The full (clean, worst-node@20%) curve over B.
- The chosen B\* and q per cell.
- Own-attack-only worst-node.
- Per-split effects.

## Declared risks

- **Gradient masking.** tanh saturates, so plain PRBCD may be weaker against CAP than against S0. BPDA attacks address this, but measured worst-node is still accuracy against the tested attacks only. Certified accuracy is the only attack-independent number, and it can't exceed MLP-M's accuracy. So no H80 result can show that CAP\* beats MLP-M against every attack; it shows this against the tested suite only.
- **Correction is a known design.** Combining an MLP with a GNN residual is common, so novelty is not claimed before a prior-art check on certified or bounded graph corrections (for example AGNNCert and randomized smoothing).
- **Scope.** Cora only; the threat model is edges only, and a text attacker reaches m directly. No low-label or other-dataset claim.
