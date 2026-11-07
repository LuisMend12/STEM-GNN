# Stage 2 analysis (Amendment 2)

Source: 15 cells (3 splits × 5 seeds, config hash `ce407cf8cd`), all `COMPLETED`, no tracebacks. Output of `STEM-GNN/scripts/llm_stem/analyze_stage2.py` is in `results_stage2/analysis_output.txt`. Paired t over 15 cells, conditional on Cora; the 5 seeds within a split share a split, so the cells aren't fully independent. Points are percentage points. Prior temperature was 2.0 in every cell. Rotations {0, 2, 4} came from the frozen audit.

## Conclusion

- **Robustness (H70–H72): no tested placement met preregistered support at worst-node 20%; S-A was inconclusive.**
  - S-F (Qwen input): −5.98 [−7.26, −4.70]. Evidence against.
  - S-V (VQ distillation): −1.81 [−2.82, −0.80]. Evidence against.
  - S-A (Qwen head prior): +0.12 [−1.26, +1.49]. Inconclusive. S-A also fails the clean guardrail at −1.77 [−3.08, −0.45].
- **Graph-free models dominate every graph arm on worst-node accuracy.** max(MLP-M, MLP-Q, QWEN) − S0 is +38.2. Under this edge-only attacker, a graph-free model is the better choice for worst-case accuracy. S0 is better on clean accuracy, by +6.94 [+5.81, +8.08] against the per-cell best graph-free model.
- **Low-label generalization (H73–H75): S-A is supported.**
  - S-A (H75): +6.98 [+4.34, +9.61], Holm p = .0002, 15/15 cells positive.
  - S-F (H73): −8.12. Evidence against.
  - S-V (H74): −0.91. Inconclusive.
- **All three staged claims for S-A on low-label accuracy are met** (one Holm family):
  - Against its placebo S-AP: +16.91 [+13.09, +20.74].
  - Against MIX: +2.45 [+0.91, +4.00], Holm p = .0043, 13/15 cells.
  - Against the per-cell max of the graph-free baselines: +10.15 [+8.97, +11.33].

## How to read the S-A low-label result

- **Mechanism.** S-A adds KL(calibrated Qwen zero-shot prior ‖ model) on unlabeled non-validation nodes, which include test nodes. This is transductive distillation of an LLM's zero-shot labels into the GNN head. The test nodes' text is used. Current-split test labels don't enter low-label training or calibration. The frozen cross-split prior audit has the overlap declared in the protocol. The Qwen prior alone scores 62.36. S-A scores 72.57, so the GNN plus prior beats both the prior (+10.2) and S0 (+7.0).
- **The placebo is weak.** S-AP, the wrong-node prior, is actively harmful: −21.78 on full-label clean accuracy against S0. So S-A − S-AP mostly measures the harm of a wrong prior, not specificity beyond a neutral prior. A uniform or label-marginal prior would be the right control, and it wasn't run.
- **Beating MIX is the more informative staged claim.** It shows that applying the prior inside training adds about 2.5 points over blending it in at prediction time, with α tuned on the same validation set.
- **Validation budget.** The low-label setting keeps the original 500-node validation set for temperature, MIX α and checkpoint selection. A realistic 21-label setting wouldn't have 500 validation labels. The effect of a small validation set on any arm wasn't tested, and the low-label gain could shrink under one.
- **Per-split.** S-A − S0 is +6.82, +5.65 and +8.46 on splits 0, 1 and 2. The effect is consistent, but the variation between splits isn't well estimated with 3 splits.
- **Trade-off with full labels.** S-A costs −1.77 clean and +0.12 worst-node. The prior helps when labels are scarce and slightly hurts when they're plentiful.
- **Descriptive.**
  - S-VA (S-V plus S-A): +8.09 low-label against S0, and +1.11 [−0.12, +2.35] against S-A. That isn't distinguishable from S-A alone, so the VQ distillation adds nothing shown.
  - MIX: +0.96 worst-node and +4.52 low-label against S0. Its α was 0–0.5 on full labels and mostly 0.5–0.75 on low labels.
- **Not computed.** The teacher diagnostic (accuracy on Qwen-correct and Qwen-wrong nodes) can't be computed for low labels. The saved per-node files hold only full-label predictions, so it would need low-label predictions regenerated from the checkpointed encoders. It isn't known whether S-A's low-label gain is concentrated on Qwen-correct nodes. That would be consistent with greater reliance on Qwen, though it wouldn't establish it.

## Novelty

Not established. Distilling LLM zero-shot or pseudo-labels into a GNN for label-scarce text-attributed graphs has close prior work, for example LLM-as-annotator and label-free node classification lines. The prior-art check hasn't been run. Until it is, this is a validated improvement to STEM-GNN in the low-label regime on Cora, not a novel method.

## Limits

Cora only, one LLM (Qwen2.5-1.5B), one λ, an edge-only attacker that sees test labels, and no text attacks: a text attacker would reach Qwen and the MLPs directly. Low-label models weren't attacked. S-A's no-VQ and uniform-prior controls weren't run.
