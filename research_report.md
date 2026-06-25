# MoE Routing Stability in STEM-GNN: Motivation, a Reproducible Failure Mode, and Tested Fixes

**Date:** 2026-06-24
**Codebase:** STEM-GNN (GNN encoder + vector-quantized codebook, optional Mixture-of-Experts encoder layers)
**Scope:** Degree-shift OOD benchmark on Cora and Pubmed, no-pretrain base case

---

## 1. Background and Motivation

STEM-GNN's encoder supports an optional Mixture-of-Experts (MoE) configuration: instead of a single weight matrix per layer, each node's representation is routed through a learned mixture of `K` expert weight matrices, selected by a per-node softmax router conditioned on the node's current-layer representation.

The motivating hypothesis for this investigation: MoE can improve out-of-distribution (OOD) generalization in graph encoders, but may introduce instability — *routing shift* (the router's behavior changing under distribution shift) and *expert imbalance* (uneven or degenerate expert usage) — that standard training does not address. The original direction was to investigate whether an LLM-derived signal could be used to stabilize MoE routing under OOD shift.

This report documents what was actually found when this hypothesis was tested empirically.

---

## 2. Experimental Setup

**Base case (no pretraining required).** STEM-GNN's normal pipeline pretrains the encoder + vector quantizer (VQ) across multiple graph domains before finetuning on a downstream task. Passing `--pretrain_dataset na` (with no `--pretrain_path`) bypasses this entirely — the encoder and VQ are randomly initialized and trained from scratch directly on the downstream task. This made rapid iteration possible without multi-day pretraining runs. (Two bugs that incorrectly blocked this no-pretrain path in the OOD evaluation scripts were identified and fixed as part of this work.)

**Benchmark.** Degree-shift OOD evaluation on node classification (`scripts/degree_shift_ood.py`), run on Cora (2,708 nodes) and Pubmed (~19,700 nodes). Test nodes are bucketed by graph degree into:
- **ID**: middle 70% of nodes by degree
- **OOD-low**: lowest 15% (low-degree, sparse-neighborhood nodes)
- **OOD-high**: highest 15% (high-degree, dense-neighborhood nodes)

**New instrumentation built.** The existing OOD scripts measured only downstream accuracy. New per-bucket MoE routing diagnostics were added, using the encoder's existing (previously unused) router-weight caching hooks:
- `entropy_of_avg`, `max_usage`: aggregate expert-usage imbalance within a bucket (entropy of the mean router distribution; fraction of nodes assigned to the most-used expert)
- `mean_node_entropy`: per-node routing decisiveness
- `top1_frac`: per-expert usage fraction
- TV distance / KL divergence between the ID bucket's and each OOD bucket's average router distribution — the direct measure of *routing shift*

**Protocol.** All results below are means ± standard deviation over **10 independent random seeds** per configuration, unless noted otherwise (one experiment used 3 seeds, flagged explicitly).

---

## 3. Result 1 — Motivation: Does MoE Actually Help?

A 2×2 ablation (MoE on/off × VQ on/off) was run on Cora to separate MoE's contribution from VQ's:

| Cora, 10 runs | ID accuracy | OOD-low accuracy | OOD-high accuracy |
|---|---|---|---|
| No MoE, VQ on | 82.57% ± 3.71 | 77.09% ± 3.58 | 80.00% ± 4.79 |
| MoE, VQ on | 87.22% ± 1.22 | 80.34% ± 0.69 | 85.54% ± 1.04 |
| No MoE, VQ off | 86.28% ± 1.49 | 79.98% ± 1.06 | 85.00% ± 0.80 |
| MoE, VQ off | 86.58% ± 1.65 | 80.05% ± 0.85 | 85.27% ± 1.00 |

**Finding:** MoE's apparent benefit is almost entirely conditional on VQ being present. With VQ on, MoE adds +4.65 points of ID accuracy; with VQ off, MoE adds only +0.30 points — within noise. Removing VQ alone (no MoE) already recovers most of MoE's apparent benefit, with *lower* variance than either VQ-on condition.

**Interpretation:** in this no-pretrain base case, an untrained/randomly-initialized VQ bottleneck measurably hurts a plain GNN's accuracy and increases run-to-run variance. MoE's measured benefit in the naive (VQ-on) comparison is largely compensating for this VQ-induced harm, not an independent contribution from the mixture-of-experts mechanism itself. This is a more precise and defensible motivation claim than "MoE improves OOD generalization."

---

## 4. Result 2 — Reliability Issue: Expert Collapse

With MoE enabled, VQ on, and no anti-collapse regularization (`lamda_env=0`, the codebase default), the new routing instrumentation reveals a severe and highly reproducible failure mode:

| Dataset, 10 runs | Layer 0 `max_usage` | Layer 1 `max_usage` | Pattern |
|---|---|---|---|
| Cora | 1.000 ± 0.001 | ≈0.45–0.90 (varies by run) | Full collapse, every seed |
| Pubmed | 0.993 ± 0.021 | 0.558 ± 0.112 | Full collapse, every seed |

**Layer 0** (router conditioned on raw node text features) **fully collapses onto a single expert in 10 of 10 seeds on both datasets.** Which expert wins varies randomly by seed, but exactly one expert wins completely every time — the other two are never selected as any node's top choice.

**Layer 1** (router conditioned on aggregated hidden states from Layer 0) does not fully collapse, but shows genuine, nonzero routing shift between ID and OOD buckets (TV distance ≈ 0.01–0.05 depending on dataset/bucket). On Pubmed this shift is asymmetric: shift toward OOD-high (TV = 0.0311 ± 0.0209) is notably larger than toward OOD-low (TV = 0.0092 ± 0.0032).

**Significance:** this is not a dataset-specific artifact. It is reproduced independently, with the same qualitative pattern (Layer 0 total collapse, Layer 1 partial imbalance + genuine shift), on two datasets with different sizes, class counts, and degree distributions.

---

## 5. Tested Fix #1 — Richer LLM-Derived Router Input (Negative Result)

**Hypothesis:** the router's input (raw node text features) lacks sufficiently stable, semantically meaningful signal, causing it to lock onto an arbitrary expert. Fix: augment the Layer-0 router's input with a per-node "semantic profile" — cosine similarity between the node's text embedding and every class's text embedding (a 7-dim vector for Cora) — derived from the same sentence-transformer embeddings already used elsewhere in the pipeline. This signal depends only on the node's own text, not its neighborhood, so it should be stable under degree shift. It does not leak the true label (similarity to *all* classes is used, not just the correct one).

| Cora | Layer 0 `max_usage` | ID accuracy |
|---|---|---|
| Random input only (10 runs) | 1.000 ± 0.001 | 87.22% ± 1.22 |
| + LLM class-similarity input (3 runs) | 1.000 ± 0.000 | 86.92% ± 0.94 |

**Result:** no effect. Collapse is total and identical with or without the richer input; accuracy is unchanged within noise.

**Conclusion:** expert collapse is not an information-availability problem. The router already had adequate signal to differentiate experts; providing additional, more stable, more semantically grounded signal did not change the outcome.

---

## 6. Tested Fix #2 — LLM-Grounded K-Means Router Initialization (Negative Result)

**Hypothesis:** collapse is a "rich-get-richer" effect driven by random initialization — whichever expert receives a small accidental early advantage captures disproportionate gradient signal, starving the others. Fix: initialize each expert pre-assigned to a distinct k-means cluster of node text embeddings (k = number of experts), instead of random weights, so experts start in genuinely different regions of input space with no arbitrary winner. (This mirrors a fix already present elsewhere in this codebase: `VectorQuantize(kmeans_init=True)` uses the same technique to prevent the analogous codebook-collapse failure mode in vector quantization.)

Implementation: router weight rows set to (normalized) cluster centroids, bias set to `-0.5 * ||centroid||^2`, so that `argmax_k(x . w_k + b_k)` exactly reproduces nearest-centroid k-means assignment at initialization, while remaining an ordinary learnable layer thereafter. Verified at epoch 0: all three experts have meaningful, non-degenerate usage (e.g. `top1_frac = [0.14, 0.42, 0.44]`).

| Cora, 10 runs | Layer 0 `max_usage` | ID accuracy |
|---|---|---|
| Random init | 1.000 ± 0.001 | 87.22% ± 1.22 |
| K-means init (balanced at epoch 0) | 0.999 ± 0.004 | 86.41% ± 2.49 |

**Result:** despite a confirmed, genuinely balanced starting point, training collapsed the router back to a single expert in 9 of 10 runs (the 10th reached 0.987, still near-total). No accuracy improvement.

**Conclusion:** collapse is not caused by where training starts. Removing the random early-asymmetry mechanism did not prevent the degenerate outcome, indicating the optimization process itself is driven toward collapse regardless of initial conditions.

---

## 7. Result 3 — Entropy Regularization Fixes Collapse (Positive Result)

Both tested fixes targeted the router's *input* or *initialization*. The remaining lever — the *loss function* — was tested next. The codebase already contains an unused regularization term, `lamda_env`, that penalizes low-entropy (collapsed) routing via an entropy-based loss on the router's output distribution; it defaults to `0.0` (effectively disabled).

| Cora, 10 runs | Layer 0 `max_usage` | ID accuracy | OOD-low | OOD-high |
|---|---|---|---|---|
| `lamda_env = 0` (collapsed) | 1.000 ± 0.001 | 87.22% ± 1.22 | 80.34% ± 0.69 | 85.54% ± 1.04 |
| **`lamda_env = 0.1`** | **0.463 ± 0.072** | 87.39% ± 1.19 | 80.25% ± 0.85 | 85.76% ± 0.69 |

**Result:** entropy regularization eliminates collapse completely. Layer 0's max expert usage drops from total collapse (1.000) to genuine three-way diversity (≈0.46, close to the theoretical balance point of 0.33 for 3 experts), in every one of 10 runs. Downstream accuracy is statistically unchanged (within noise) — the fix has no measured cost. ID→OOD routing shift in Layer 0 also shrinks substantially (TV ≈ 0.0010–0.0012, down from the larger Layer-1 shifts observed without regularization).

---

## 8. Discussion: What the Three Results Together Establish

The two negative results and the one positive result triangulate the cause of collapse:

1. Richer input (negative) rules out "the router lacks adequate information."
2. Balanced initialization (negative) rules out "collapse is caused by random early asymmetry."
3. Entropy regularization (positive) confirms the cause is the **loss landscape / optimization dynamics**: with no penalty for low entropy, gradient descent is driven toward the degenerate, fully-collapsed solution regardless of input richness or starting point. A direct penalty on that outcome is sufficient to prevent it, at no accuracy cost.

**Consequence for the original LLM-stabilization hypothesis:** expert collapse, as a standalone problem, is not an open research question for this architecture — it has a known, free, already-implemented fix. An LLM-based contribution aimed at "fixing collapse" would be solving an already-solved problem. The genuinely open target is the **residual ID→OOD routing shift that remains after applying entropy regularization** — smaller than the unregularized case, but not yet shown to be statistically indistinguishable from zero, and shown to be asymmetric across OOD directions on Pubmed even without regularization. This is a narrower, sharper, and more defensible target for an LLM-grounded mechanism than the original framing.

---

## 9. Limitations

- `lamda_env = 0.1` was a single chosen value, not swept; its sensitivity is untested.
- The residual routing-shift magnitude after regularization (TV ≈ 0.001–0.002) has not been tested for statistical significance against sampling noise.
- The `lamda_env` fix has only been confirmed on Cora; generalization to Pubmed (where collapse itself was independently confirmed) has not yet been tested.
- Whether the "winning" expert in a collapsed run corresponds to any meaningful semantic or structural grouping of nodes has not been investigated.
- All results are on the degree-shift benchmark only; the codebase's homophily-shift, missing-feature, and edge-drop OOD benchmarks have not yet been run with this routing instrumentation.
- The richer-input and k-means-init experiments were tested only on Cora (the k-means-init result at 10 runs, the richer-input result at 3 runs).

---

## 10. Next Steps

1. Confirm the `lamda_env` fix generalizes to Pubmed, using the same 10-seed protocol used to confirm collapse itself.
2. Statistically characterize the residual routing shift after regularization — is it distinguishable from noise? Does it grow under more severe distribution shift (homophily-shift, missing-feature benchmarks)?
3. If the residual shift is real, design and test an LLM-grounded mechanism targeting it specifically — e.g., a routing-consistency loss across augmented/perturbed node views, anchored by LLM-derived semantic similarity — layered on top of `lamda_env`, not replacing it.
4. Extend the routing instrumentation built here to the remaining OOD benchmark scripts (homophily-shift, missing-feature, random-edge-drop).
5. Sweep `lamda_env` to characterize the accuracy/diversity tradeoff curve, rather than relying on a single tested value.

---

## Appendix: Reproducibility

All experiments used `scripts/degree_shift_ood.py` with `--use_params --finetune_dataset {cora|pubmed} --pretrain_dataset na --repeat 10` (no-pretrain base case), varying:

- Baseline (no MoE): no `--moe` flag
- MoE: `--moe --moe_layers all --moe_experts 3 --moe_tau 1.0`
- VQ off: add `--use_vq 0`
- Richer router input: add `--use_aux_router`
- K-means router init: add `--router_init kmeans`
- Entropy regularization: add `--lamda_env 0.1`

New code added to the repository as part of this work:
- Per-bucket MoE routing diagnostics in `scripts/degree_shift_ood.py` (`compute_routing_cache`, `compute_bucket_routing_stats`, `compute_routing_shift`)
- `--use_aux_router` (class-similarity router input) in `model/encoder.py`, `model/ft_model.py`, `task/node.py`, `utils/others.py`
- `--router_init kmeans` (`Encoder.init_router_kmeans`) in `model/encoder.py`
- Bug fixes: corrected directory-depth path bugs and a no-pretrain-bypass bug that previously prevented `--pretrain_dataset na` from working in `degree_shift_ood.py`, `homophily_shift_ood.py`, and `tri_objective.py`
