# Analysis: Text-Anchored Edge Floor (TAEF) against message collapse

Reviewed by Codex (`../reviews/codex_moe_depth_collapse_review_2026-10-22.md`). Its corrections are applied and each was verified against the raw JSON.

**Setup.** STEM-GNN's real `Encoder` (BatchNorm, hidden width 64), OFA Cora.

**Confirmatory run:** `results/split*_multiseed.json`, with 5 seeds × 3 splits per cell (n = 15, paired by split and seed) and threads fixed at 4. This is a prospectively specified multi-seed follow-up, run *after* earlier test results were inspected. The original settings (ρ, λ, methods) were retained, but the study is not wholly outcome-blind. Amendment timing is documented in the protocol; it is not independently established here.

Standard deviations are sample SDs. Paired SEs are the SD of the 15 matched differences divided by √15. All tabulated values match the raw JSON.

## Results (test accuracy %, mean ± SD over 15 runs; effective rank in parentheses)

| Encoder | Depth | Baseline | TAEF (ρ=0.5) | GEF (EGNN-style) | TAEF ρ=0.25 [EXPL] |
|---|---|---|---|---|---|
| SAGE | 2 | 76.2±3.9 (29.9) | 73.1±4.5 (34.2) | 75.4±3.7 (30.9) | 74.7±4.4 (31.6) |
| SAGE | 16 | 74.5±3.1 (8.2) | 74.4±2.1 (8.7) | 74.8±2.4 (9.0) | 74.8±2.3 (8.5) |
| MoE | 2 | 80.0±1.9 (24.2) | 78.7±2.6 (27.9) | 79.6±1.9 (25.1) | 79.7±2.1 (25.2) |
| MoE | 16 | 74.8±2.6 (9.0) | 73.5±4.1 (9.1) | 73.3±3.2 (9.4) | 74.3±3.7 (9.4) |

Collapse metrics for the baseline, depth 2 → 16:

| Encoder | Global MAD | Edge-distance ratio | Effective rank |
|---|---|---|---|
| SAGE | 0.609 → 0.952 | 0.200 → 0.143 | 29.9 → 8.2 |
| MoE | 0.911 → 0.935 | 0.235 → 0.153 | 24.2 → 9.0 |

Paired differences (mean, SE):

| Comparison | SAGE d2 | SAGE d16 | MoE d2 | MoE d16 |
|---|---|---|---|---|
| TAEF − baseline | −3.09 (0.94) | −0.17 (0.97) | −1.32 (0.27) | −1.23 (1.09) |
| GEF − baseline | −0.83 (0.17) | +0.23 (0.90) | −0.38 (0.14) | −1.52 (0.71) |
| TAEF − GEF | −2.26 (0.85) | −0.40 (0.92) | −0.93 (0.25) | +0.29 (0.81) |

**Baseline depth drop:** SAGE −1.67 (SE 1.10); MoE −5.22 (SE 0.62). MoE − SAGE drop difference: −3.55 (paired SD 3.93, SE 1.01).

## Hypothesis outcomes

The threshold outcomes below are descriptive point-estimate results. Amendment 2 specifies paired means, but it does not explicitly supersede the original "within one SD is no detectable difference" rule or say which SD applies. So threshold failure and detectable effects are reported separately, and no retrospective switch from SD to SE is assumed.

| ID | Point-estimate outcome | Detectability note |
|---|---|---|
| **H15** collapse ≥ 5 for non-MoE | **Not supported:** SAGE −1.67, and the registered conjunction fails (MAD *increases*) | — |
| **H16** MoE resists collapse better | **Not supported. Means are reversed** (MoE drops 3.55 more) | The reversal is within one paired SD (3.93), so a detectable reversal is not established under the original rule. |
| **H17** TAEF recovers ≥ 3 at depth 16 | **Not supported:** −0.17 (SAGE), −1.23 (MoE) | Detectable harm is not established for either encoder. |
| **H18** TAEF beats GEF by ≥ 1.5 | **Refuted on the point estimate:** encoder-averaged −0.05 (SE 0.61) | No detectable difference. No evidence that TAEF is meaningfully worse than GEF. |
| **H19** TAEF is free at depth 2 | **Fails the ±1 criterion:** −3.09 (SAGE), −1.32 (MoE) | Not a formal equivalence test. |

## What we learned

1. **Depth changes local structure, not global spread.** Depth reduces centered effective rank and mean edge cosine distance relative to the input. Global random-pair MAD does not approach zero: it rises from 0.609 to 0.952 for SAGE and from 0.911 to 0.935 for MoE. That is spectral concentration and local smoothing, without global angular convergence. A distance floor does not target rank.
2. **Neither energy floor recovers depth-16 accuracy.** That holds for the text-anchored per-edge floor and the graph-level floor alike. At depth 2, TAEF lowers mean accuracy (SAGE −3.1 points).
3. **Caveats on the metrics:**
   - They do not establish that rank loss *causes* accuracy loss, or that TAEF harms useful smoothing.
   - The "energy ratio" is mean edge cosine distance, not conventional squared-distance Dirichlet energy.
   - Effective rank uses centered states and normalized singular values. The implementation needs a zero-spectrum guard for exact collapse.
   - High average router-usage entropy does not rule out node-level specialization or collapse in an individual layer.
4. **Methodology:** single-seed CPU runs are unreliable here. The protocol records the same seed giving 76.4 vs 68.9 at 4 vs 8 threads; the supplied files do not establish that thread count alone caused it. Separately, deep-MoE runs are not reproducible even at a fixed thread count (see `../moe-depth/analysis.md`).

## Threats to validity

- One homophilous dataset.
- Width 64.
- No pretraining.
- Only ρ ∈ {0.25, 0.5} and λ = 1 were tried.
- A rank-targeting regularizer was not tested.
