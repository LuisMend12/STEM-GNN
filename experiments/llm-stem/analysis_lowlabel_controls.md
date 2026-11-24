# Analysis: controls for the low-label Qwen-prior result

Protocol: `lowlabel_controls_protocol.md` (written and Codex-reviewed before the run). Code:
`STEM-GNN/scripts/llm_stem/lowlabel_controls.py`, `analyze_lowlabel_controls.py`. Raw output:
`lowlabel_controls_output.txt` (run on the cluster, Slurm 33088290). Runs: Slurm array 33088080, all 15 cells
completed with no errors, one code/audit/prior/environment identity. The selfcheck (Slurm 33087843) passed.
Codex results review: `../reviews/codex_lowlabel_controls_results_review_2026-11-09.md` (all findings
applied).

**Reproduction.** The V500 JSON accuracies for S0, S-A, MLP-M and QWEN exactly match Stage 2 in every cell
(max drift 0.000 points). The cluster analysis completed and asserts that the saved predictions reproduce
every reported accuracy. Prediction files stay on the cluster, so prediction-level checks and the
conditional-accuracy diagnostics can't be recomputed from this repository alone.

## Bottom line

- **With V500, S-A outperforms both tested control recipes.**
  - It beats uniform-target distillation by +10.68 points and the specified MiniLM-MLP distillation
    recipe by +7.03. Both win in 15/15 cells and are Holm-significant (H90, H91 supported).
  - Descriptively, S-U and S-T differ from S0 by −3.70 and −0.05.
  - This supports superiority over these two controls, not a general claim about Qwen-specific
    information, other soft targets or other text teachers.
- **The strict-budget claim is not met.** The strict arm S-A0 uses Qwen's rotation-0 prompt with no audit.
  With 21 training + 21 validation labels, its comparisons against S0 (−0.97 [−4.86, +2.91]), uniform
  (+2.37) and the MiniLM teacher (+2.27) are all inconclusive (H92–H94).
- **Descriptively, the rotation-averaged prior does better than rotation 0 in both validation regimes.**
  - Standalone accuracy: 62.4% versus 53.5%.
  - Under V21, S-A beats S0 by +5.66 [+2.25, +9.08], positive in 12/15 cells, but S-A itself loses 3.95
    points relative to V500.
  - This is consistent with prior construction contributing to the difference. It doesn't establish
    prior quality as the only explanation, or rule out validation effects.
  - S-A inherits the audit's use of 30 labels, some of them test nodes in other splits.

## Confirmatory results (one Holm family of five; n = 15; run-level, conditional on Cora)

| ID | Comparison | Regime | Mean [unadjusted 95% CI] | Wins | Holm p | Verdict |
|---|---|---|---|---|---|---|
| H90 | S-A − S-U | V500 | +10.68 [+7.45, +13.91] | 15/15 | <.0001 | Supported |
| H91 | S-A − S-T | V500 | +7.03 [+3.98, +10.08] | 15/15 | .0009 | Supported |
| H92 | S-A0 − S0 | V21 | −0.97 [−4.86, +2.91] | 5/15 | .72 | Inconclusive |
| H93 | S-A0 − S-U | V21 | +2.37 [−1.78, +6.53] | 7/15 | .72 | Inconclusive |
| H94 | S-A0 − S-T | V21 | +2.27 [−2.69, +7.22] | 7/15 | .72 | Inconclusive |

- **Claim A** (Qwen prior beats the uniform target, full validation): supported. This is narrower than
  "node-specific Qwen content matters", which would need an entropy-matched shuffled-target control.
- **Claim B** (gain survives a strict 21 + 21 label budget): not met.
- **Superiority over this MiniLM-MLP teacher:** supported in V500; not established in V21.

## Low-label test accuracy (mean ± sd over 15 cells, points)

| Arm | V500 | V21 |
|---|---|---|
| S0 | 65.59 ± 4.86 | 62.96 ± 4.40 |
| S-A (Qwen, rotations {0, 2, 4}) | 72.57 ± 2.14 | 68.62 ± 4.61 |
| S-A0 (Qwen, rotation 0) | 64.24 ± 3.28 | 61.98 ± 4.62 |
| S-U (uniform) | 61.89 ± 5.66 | 59.61 ± 6.73 |
| S-T (MiniLM-MLP teacher) | 65.54 ± 5.58 | 59.72 ± 7.81 |
| MIX | 70.11 ± 2.10 | 64.43 ± 4.76 |
| QWEN (prior alone, rotations {0, 2, 4}) | 62.36 ± 0.38 | 62.36 ± 0.38 |
| QWEN0 (prior alone, rotation 0) | 53.48 ± 0.40 | 53.48 ± 0.40 |
| MLP-M (teacher alone) | 54.22 ± 5.18 | 51.32 ± 7.04 |

## Descriptive

- **Each arm against S0** (paired, uncorrected):
  - V500: S-A +6.98, MIX +4.52, S-T −0.05, S-A0 −1.35, QWEN −3.23, S-U −3.70.
  - V21: S-A +5.66 [+2.25, +9.08] (12/15), MIX +1.47, S-A0 −0.97, S-T −3.24, S-U −3.35, QWEN −0.59.
- **S-A − S-A0 under V21:** +6.64 [+3.98, +9.30], 14/15.
- **Validation-regime differences (V21 − V500):**
  - S0 −2.63, S-A −3.95, S-A0 −2.26, S-U −2.28, S-T −5.82, MIX −5.69, MLP-M −2.90.
  - Every trained arm's mean accuracy drops.
  - For the KL arms, these differences combine two changes: less validation information, and the other
    479 validation nodes joining the unlabeled anchor set.
  - QWEN and QWEN0 predictions don't change, because temperature scaling preserves the argmax.
- **Temperatures and α:** the Qwen temperature was 2.0 for both priors in V500, and 1.0–4.0 in V21. MIX α
  ranged 0–0.75 in V500 and 0–0.5 in V21.
- **Student accuracy conditional on teacher correctness** (pooled node-cell observations):
  - S-A scores 87.7% where Qwen is correct and 47.4% where it's wrong (V500).
  - S-T scores 87.0% and 40.1% on its own teacher's corresponding subsets.
  - These subsets differ between teachers, so the rates can't explain why S-A beats S-T.
  - Shared node difficulty is a possible explanation: S0 already scores 85.4% / 42.2% on the MLP
    teacher's subsets.
  - Matched-subset comparisons against S0, and student–teacher agreement (including agreement on wrong
    labels), would help assess reliance on the teacher, but wouldn't alone establish causality.
- **Per split:**
  - V500: S-A − S-U is +9.26 / +13.06 / +9.73; S-A − S-T is +9.05 / +3.42 / +8.62.
  - V21: S-A0 − S0 is +0.68 / −5.47 / +1.88.
  - The supported effects keep their sign in every split; the strict-arm effect doesn't.

## Interpretation

1. **The Stage 2 result survives the two specified controls under V500.** Neither uniform-target
   regularization nor this MiniLM-MLP distillation recipe reproduces S-A's improvement. Entropy-matched
   controls, and stronger or differently tuned text teachers, remain untested.
2. **Prior construction is a candidate explanation.**
   - S-A's descriptive gain over S0 is +6.98 in V500 and +5.66 in V21; S-A0's corresponding differences
     are −1.35 and −0.97.
   - These comparisons change the prior logits and, in V21, can also change the selected temperature.
     They don't isolate standalone teacher accuracy as the mechanism.
   - Rotation averaging combines predictions across option orderings. This experiment doesn't directly
     measure whether it reduces answer-position bias.
   - The audit used 30 labeled nodes to choose between rotation 0 and the fixed {0, 2, 4} subset. Its
     decision statistic was the gain from averaging all seven rotations. A pipeline that averages
     without the audit's labels (temperature still chosen on the designated validation set) hasn't been
     tested independently.
3. **Next test.** Before looking at a second dataset's outcomes, preregister:
   - a rule that doesn't depend on the number of classes, for example average the class-aligned logits
     over every distinct cyclic option rotation, then choose the temperature only on the designated
     validation labels;
   - three training and three validation labels per class (3C + 3C labels for C classes);
   - the comparisons, thresholds, multiplicity correction and sample design.

   That would test a newly specified pipeline on another dataset, not literally replicate Cora's {0, 2, 4}
   subset. More evaluation on the already-inspected Cora splits can't give independent confirmation.

## Limitations

- Cora only. The intervals are run-level: 5 seeds share each split and all cells share one graph, and
  Holm doesn't correct for that dependence.
- S-A in V21 is descriptive (not in the Holm family) and inherits the audit's label use.
- S-U is uniform-target cross-entropy (KL(U ‖ p)), not an entropy-matched control. S-T is one untuned
  MiniLM-MLP recipe, and MiniLM is itself a pretrained language encoder.
- V21 selection is coarse (accuracy steps of 1/21) and reuses the same 21 nodes for checkpoint,
  temperature and α.
- This is transductive evaluation: test-node text and teacher targets enter KL training. It says nothing
  about unseen nodes or graphs.
- The label budget counts the training and validation labels kept under class-balanced benchmark
  sampling. Building these subsets uses class labels in the original pools; getting such balanced
  subsets from initially unlabeled nodes could need extra labeling.
- Only one Qwen model, one fixed KL weight and one MiniLM-MLP teacher recipe were tested.
- No novelty claim: LLM pseudo-label distillation for graphs with scarce labels has prior work.
