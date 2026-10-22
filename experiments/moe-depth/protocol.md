# Protocol: why STEM-GNN's MoE encoder degrades with depth

Written 2026-10-22, before any result for these hypotheses existed.

## Motivation

`experiments/message-collapse` used 15 paired runs. From depth 2 to 16, the MoE encoder lost **−5.2 ± 0.6** points while plain SAGE lost only −1.7 ± 1.1. This happened even though the MoE encoder has residuals and there is no expert collapse (router entropy 0.92). Both encoders end with an effective rank of about 9.

ARS failure mode 5 says a surprising first result must be explained mechanistically before it is treated as a finding.

## Candidate mechanism

**Train/eval routing mismatch compounding with depth.** `model/encoder.py:427-433` routes with `F.gumbel_softmax(logits, tau)` in training and `F.softmax(logits)` at eval. Each MoE layer therefore sees a different expert mixture at test time than during training. With 16 stacked MoE layers, small per-layer mismatches can compound.

## Quick prior-art pass (2026-10-22, two web searches; not a full check)

**Found:** graph MoE with noisy top-k gating (GMoE, NeurIPS 2023), soft-to-hard routing in sparse MoE (arXiv:2605.02124, LLM MoE), and Gumbel-softmax used for graph rewiring (arXiv:2508.17531).

**Not found:** work linking Gumbel router train/eval mismatch to depth degradation in MoE GNNs. Treat novelty as unverified.

## Variants (MoE: real `Encoder`, `moe_layers='all'`, K=3, τ=1)

| ID | Training routing | Eval routing | Purpose |
|---|---|---|---|
| **V0** | Gumbel-softmax (repo default) | softmax | baseline |
| **V1** | — (the same trained V0 model) | **mean logits over 16 Gumbel samples**, with noise injected only into router logits; dropout and BatchNorm stay in eval mode | removes the mismatch at eval time, no retraining |
| **V2** | **softmax (no Gumbel noise)** | softmax | removes the mismatch at train time |
| **V3** | **uniform 1/K, router frozen at zero logits, no noise** | uniform | MoE layer architecture with no routing |
| **SAGE** | plain SAGE encoder | — | reference |

**Mechanics of V2 and V3.** `encoder.py`'s module-level `F` is replaced, for these variants only, by a shim whose `gumbel_softmax(l, tau)` returns `softmax(l / tau)`. In V3 the router weights are zeroed and frozen.

**Shared setup:** the same harness as `message-collapse`.

- BatchNorm, dropout 0.15, hidden width 64, Adam (lr 0.01, wd 5e-4).
- Up to 300 epochs, patience 50.
- **Threads fixed at 4.** Seeds {0..4} × splits {0, 1, 2}; seed = 1000·seed + split.
- Depths {2, 16}.

## Metrics

- Test accuracy.
- **Routing disagreement:** for the V0 model, the mean total-variation distance between eval softmax weights and Gumbel-sampled weights, per layer.

## Hypotheses (confirmatory; paired by split and seed, n = 15)

All "drop" quantities are depth-16 accuracy minus depth-2 accuracy.

| ID | Hypothesis | Support if | Refuted if |
|---|---|---|---|
| **H20** | The mismatch costs accuracy at depth 16 | V1 − V0 at depth 16 ≥ +1.5 | ≤ 0 |
| **H21** | Removing Gumbel noise from training shrinks the depth drop | drop(V2) − drop(V0) ≥ +3 | < +1 |
| **H22** | Routing, not the MoE layer architecture, is responsible | drop(V3) − drop(V0) ≥ +3 | drop(V3) ≤ drop(V0) + 1, meaning the architecture (residual + BN + un-ReLU'd messages) is responsible instead |
| **H23** | Effects are depth-specific | V1 − V0, V2 − V0 and V3 − V0 at depth 2 are each within ±1 | — |

## Threats to validity

- One dataset.
- Width 64.
- No pretraining.
- BatchNorm on.
- τ = 1 only.
- V1 averages logits, not probabilities.

## Amendment (2026-10-22, before any usable result)

The first launch had an operator-precedence bug in V1's Gumbel noise: `-log(u).clamp_min(...)` clamped before negating, which produced NaN logits. That gave V1 accuracy 0.07 and NaN routing TV. It was caught on the first rows, then fixed and verified: Gumbel-max frequencies of 0.599/0.299/0.102 against the expected 0.6/0.3/0.1, and V1 = 82.6 vs V0 = 82.5 at depth 2. The partial results were deleted and the run relaunched. No hypotheses or variants changed.
