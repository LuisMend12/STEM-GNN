# Protocol: robustness of STEM-GNN's VQ finetune path under adaptive attack

Written 2026-10-20, before any result for these hypotheses existed. Prior evidence comes from `STEM-GNN/scripts/adaptive_robustness/partial_results_6models.json`, which was collected 2026-10-17/18 on 3 splits with 6 of the 11 models complete.

Prior evidence (Cora, accuracy %, mean±std, 3 splits):

| model | clean | PRBCD 5% | PRBCD 10% | PRBCD 20% |
|---|---|---|---|---|
| MLP | 72.5 | 72.5 | 72.5 | 72.5 |
| SAGE | 81.3±0.6 | 70.7±0.6 | 64.2±0.7 | 54.9±0.6 |
| SAGE+DropEdgeCons | 82.9±1.3 | 73.9±2.4 | 68.8±2.5 | 61.0±4.0 |
| SAGE+TextGate | 79.1±0.4 | 70.7±0.7 | 66.0±0.7 | 59.9±0.7 |
| SAGE+PropSens (coef 1) | **31.9±1.6** (collapsed) | 31.4 | 31.2 | 31.0 |

## Setting

**Data.** OFA Cora with frozen 384-d sentence embeddings, OFA splits 0/1/2, CPU.

**STEM-lite model.** A stand-in for STEM-GNN's finetune path. The full pipeline cannot run here because there are no pretrain checkpoints.

- **Encoder:** 2-layer weighted-mean SAGE, hidden and output width 128.
- **VQ:** cosine VQ with 4 heads × 32 dims and 32 codes per head. The codebook is initialized by spherical k-means on stage-1 embeddings and frozen during finetuning, mirroring `--freeze_vq 1`. Training uses a straight-through estimator and commitment weight 0.25, matching the repo defaults.
- **Decoder:** one linear decoder per head on the quantized codes, with logits averaged over heads (`separate_decoder_for_each_head=True`).
- **Training stages:**
  1. Encoder plus continuous per-head decoder, trained with cross-entropy. This stands in for pretraining.
  2. k-means codebook.
  3. Finetune the encoder and decoder with the codebook frozen, early-stopping on validation accuracy.

**Threat.** Global evasion PRBCD on the test nodes at budgets of 5%, 10% and 20% of undirected edges. The model is trained on the clean graph.

## Hypotheses and decision rules

**H1 — Gradient masking (measurement).** `model/vq.py` applies the straight-through estimator only when `self.training`. In eval mode the quantized output therefore has no gradient path to the encoder, so a white-box attack on STEM in eval mode is blind.

- *Prediction:* PRBCD with hard VQ will report higher accuracy than either (a) PRBCD with an STE surrogate (BPDA: forward identical, gradients straight-through) or (b) a transfer attack from a SAGE surrogate.
- *Support if:* hard-VQ PRBCD@10% − min(STE, transfer)@10% ≥ 5 points.
- *Refuted if:* the gap is < 2 points.

**H2 — DropEdge consistency transfers to STEM.** Add the prediction-consistency loss between two 30% DropEdge views during stage 3 (coef 1).

- *Support if:* STE-PRBCD@10% improves by ≥ 3 points over STEM-lite and the clean drop is ≤ 1 point.

**H3 — Code margin.** VQ maps a perturbed state back to the same code when that state stays inside the code's Voronoi cell. A hinge loss on the cosine margin between the nearest and second-nearest code, relu(0.1 − (s1 − s2)) with coef 1 and averaged over heads, should enlarge those cells' effective radius.

- *Prediction:* a lower code-change rate under STE-PRBCD@10%, and higher robust accuracy.
- *Support if:* STE-PRBCD@10% improves by ≥ 3 points and the code-change rate falls.

**H4 — Why PropSens collapses, and a repair.** The per-node relative penalty ‖h_i − h'_i‖ / ‖h_i‖ (with a detached denominator floored at 1e-6) is exactly zero for dead-ReLU nodes. That makes the all-zero hidden state an attractor.

- *Diagnostic:* the fraction of exactly-zero hidden rows after training, with coef 1, should be high.
- *Repairs to test on SAGE:*
  - (a) coef 0.1;
  - (b) a global denominator, mean_i ‖h_i‖, detached;
  - (c) the penalty applied to the pre-activation of layer 1.
- *Support if:* a repair keeps clean accuracy ≥ SAGE − 1 and reduces the zero-row fraction. Its robustness effect is a secondary, declared measurement.

## Metrics

- Clean accuracy.
- Accuracy under PRBCD-hard (STEM only), PRBCD-STE, and transfer (from SAGE's PRBCD graph) at each budget.
- Code-change rate: the fraction of (test node, head) pairs whose code changes under STE-PRBCD@10%.
- Zero-row fraction (H4).
- Report mean±std over 3 splits, and runtime.

## Confirmatory versus exploratory

H1–H4 and their decision rules above are confirmatory. Anything else is exploratory.

## Threats to validity, declared up front

- STEM-lite has no multi-domain pretraining and no MoE router.
- The data is a single small dataset.
- PRBCD runs at reduced CPU scale (block 100k, 100 epochs).
- Evasion only.
- The text encoder is MiniLM-384, not STEM-GNN's 768-d encoder.
