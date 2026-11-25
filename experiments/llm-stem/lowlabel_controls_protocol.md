# Protocol: controls for the low-label Qwen-prior result (S-A)

Written 2026-11-09, before any run. Revised after the Codex design review
(`../reviews/codex_lowlabel_controls_design_review_2026-11-09.md`), still before any run. Code:
`STEM-GNN/scripts/llm_stem/lowlabel_controls.py`, analysis `analyze_lowlabel_controls.py`. This is
direction 1 of `../generalization-frontiers/proposal.md`.

## Question

Stage 2 (`analysis_stage2.md`) found that S-A, a KL pull toward a calibrated Qwen zero-shot prior on
unlabeled nodes, raises Cora accuracy with 3 labels per class by +6.98 points over S0. Three things could
explain or undermine that:

1. **A generic soft target might do as well.** With 21 labels, a KL term on unlabeled nodes may help as a
   regularizer, whatever the target says.
2. **A non-Qwen text teacher might do as well.** The gain may come from distilling a text classifier into
   the GNN, not from the LLM specifically.
3. **Validation was unrealistically large.** Stage 2 used the full 500-node validation set for prior
   temperature, MIX α and checkpoint selection. The prior's option rotations ({0, 2, 4}) were also chosen
   by an audit on 30 validation nodes pooled over splits 0–2. Some of those can be test nodes in another
   split.

## Setting

- Same 15 cells (splits 0–2 × seeds 0–4, base = 1000·seed + split), same low-label training mask (3 per
  class, `stage2.low_label_mask`), same test mask.
- **No pretraining:** each cell reuses the label-free S0 encoder and VQ from the Stage 2 checkpoint with
  tag `ce407cf8cd`. The checkpoint's and result file's configuration must equal Stage 2's, and the audit
  file must pass Stage 2's own text-hash and rule checks with the same hash, or the cell fails before
  training.
- No attacks: clean low-label test accuracy only.
- The finetune recipe, λ = 1 and the KL direction (KL(target ‖ model)) are Stage 2's.

## Validation regimes

| Regime | Validation for prior temperatures, teacher early stopping and temperature, checkpoint selection, MIX α | KL (anchor) nodes |
|---|---|---|
| V500 | Stage 2's 500 validation nodes | Not in train or V500 (as Stage 2) |
| V21 | 3 per class drawn from validation (seeded, base + 80 001) = 21 nodes | Not in train or V21 |

- In V21, the other 479 validation nodes become ordinary unlabeled nodes, and no arm reads their labels.
- The strict V21 arm (S-A0, below) uses no audit, so its whole pipeline sees **21 training + 21 validation
  labels**.
- Test nodes are in the anchor set in both regimes (transductive, as in Stage 2). Their text is used;
  their labels are never read by the S-A0 pipeline.

## Arms (each trained in both regimes)

| Arm | KL target on anchor nodes |
|---|---|
| S0 | None |
| S-A | Calibrated Qwen prior, rotations {0, 2, 4} (Stage 2's S-A; the rotation choice came from the audit) |
| S-A0 | Calibrated Qwen prior, rotation 0 only: no audit; the strict-budget arm |
| S-U | Uniform 1/C. With KL(U ‖ p) this is uniform-target cross-entropy regularization. It is *not* the entropy confidence penalty KL(p ‖ U), and it doesn't match the Qwen target's entropy |
| S-T | Calibrated predictions of a MiniLM-MLP teacher: the Stage 2 MLP recipe on the same 21 training labels, early-stopped and temperature-scaled on the regime's validation. It uses no Qwen, but MiniLM is itself a pretrained language encoder |
| MIX | S0 mixed with the S-A prior, α on the regime's validation (descriptive) |
| QWEN, QWEN0, MLP-M | The teachers alone (descriptive) |

A training-label-marginal prior would equal the uniform one here, because the training mask is
class-balanced, so it isn't run.

## Hypotheses (one Holm family, five tests)

All tests: n = 15 cells, two-sided paired t, 95% CI, points of low-label test accuracy. The intervals are
run-level and conditional on Cora: five seeds share each split, and all cells share one graph. Holm
corrects for multiplicity, not for this dependence; split-level means are reported as a sensitivity check.

| ID | Comparison | Regime | Support | Evidence against |
|---|---|---|---|---|
| H90 | S-A − S-U | V500 | ≥ +2, Holm p < .05 | Upper CI < +1 |
| H91 | S-A − S-T | V500 | ≥ +2, Holm p < .05 | Upper CI < +1 |
| H92 | S-A0 − S0 | V21 | ≥ +2, Holm p < .05 | Upper CI < +1 |
| H93 | S-A0 − S-U | V21 | ≥ +2, Holm p < .05 | Upper CI < +1 |
| H94 | S-A0 − S-T | V21 | ≥ +2, Holm p < .05 | Upper CI < +1 |

Thresholds are on the point estimate, together with Holm p < .05.

**Claims:**
- **A (H90):** with the full validation set, the Qwen prior beats the uniform target. This is narrower than
  "node-specific Qwen content matters", which would need an entropy-matched, shuffled-target control.
- **B (H92 and H93):** the Qwen-prior gain survives a strict 21 + 21 label budget, beating both S0 and the
  uniform target.
- **Superiority over this MiniLM-MLP teacher:** H91 (V500) and H94 (V21), each reported separately. A
  non-supported result reads "superiority over the specified MiniLM-MLP teacher was not established", not
  "it is just text-teacher distillation".

**Descriptive:**
- Each arm − S0 per regime.
- S-A − S-A0 and S-A − S0 under V21.
- V21 − V500 per arm.
- Temperatures and MIX α.
- Student accuracy on teacher-correct versus teacher-wrong test nodes (pooled counts, empty subsets
  reported).
- Per-split means.

## Integrity checks

- **Reproduction.** V500's S0, S-A, MLP-M and QWEN rebuild Stage 2's low-label models from the same
  encoder, mask, seeds and calibration. Test accuracy is compared with Stage 2's `low_clean`, and the
  maximum drift is reported. Every comparison above is within this run, so drift changes no hypothesis,
  but drift above 0.5 points is flagged.
- **Analysis** requires one code, Stage 2 tag, audit, text and environment identity across all cells, and
  the exact arm × regime × cell grid.
- **Selfcheck before launch** (debug job on the cluster):
  - the Stage 2 checkpoint loads with the expected configuration;
  - the V21 mask has 3 per class inside validation and is repeatable;
  - the low-label mask is disjoint from V21 and test;
  - the S-A0 prior differs from the S-A prior;
  - the anchors exclude each regime's labels;
  - all targets are distributions;
  - the teachers ignore labels outside train and V21, given fixed masks;
  - every KL arm changes the logits (not necessarily the argmax) relative to S0;
  - V21 training is repeatable;
  - full split 0 / seed 0 rows, with the Stage 2 drift printed.

## Declared limitations

- S-A, unlike S-A0, inherits the audit's label use (30 pooled validation nodes, some of them test nodes in
  other splits). Only S-A0 supports the strict-budget claim.
- V21 selection is noisy: accuracies move in steps of 1/21, so MIX α ties go to the smaller α, and
  checkpoint and temperature selection reuse the same 21 nodes.
- S-T is one untuned teacher recipe, so a general "LLM beats text teachers" claim isn't supported by any
  result here.
- Cora only. A second dataset is the next step, whatever the result.
- Novelty isn't claimed: LLM pseudo-label distillation for graphs with scarce labels has prior work.
