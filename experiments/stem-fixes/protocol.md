# Protocol: two architecture fixes for STEM-GNN (X1 MoE initialization, X2 learnable codebook)

Written 2026-10-26, before any result. Codex will review the design before anything is run.

**Scope (user instruction):** architecture-level only. Nothing here uses text embeddings to weight or select graph structure. The node features are only the input.

Both defects are verified in the repo code (`../architecture-review/findings.md`, items 1 and 2).

## Shared setup

- STEM-GNN's real `model/encoder.py::Encoder` and real `model/vq.py::VectorQuantize`.
- OFA Cora (384-d node features, 10 provided splits); we use splits {0, 1, 2} × seeds {0..4}, so n = 15 paired runs.
- Threads 4, with `torch.use_deterministic_algorithms(True, warn_only=True)`.
- Seed = 1000·seed + split, reset before each run.
- Prior work showed deep MoE is not reproducible across processes, so seed-matched runs are treated as paired *draws*, not replications.

## X1. Per-expert initialization of MoE layers

**Defect.** `MixtureSageLayer.reset_parameters` applies `xavier_uniform_` to the stacked `[K, 2d, d]` tensor, giving variance 2/[d(2d+K)] instead of 2/(3d) per expert matrix. At d=768 that is 512× smaller (measured). Each equal-width MoE layer therefore starts as approximately its residual (identity) path.

**Variants** (`moe_layers='all'`, K=3, τ=1, BatchNorm, dropout 0.15, width d=128; at d=128 the ratio is (2d+K)/3 ≈ 86×):

| ID | Description |
|---|---|
| **M-stack** | repo initialization |
| **M-expert** | `xavier_uniform_` applied to each `weights[k]` (shape `[2d, d]`) separately; nothing else changes |
| **SAGE** | non-MoE reference |

**Depths:** {2, 16}. **Training:** linear head, Adam (lr 0.01, wd 5e-4), up to 300 epochs, patience 50 on validation accuracy.

**Diagnostics:**

- at initialization, per-layer RMS of the transform `combined @ W_k` vs the residual `x`;
- the same ratio after training;
- router entropy.

**Hypotheses (paired by split and seed):**

| ID | Hypothesis | Support if | Refuted if |
|---|---|---|---|
| **H27** | The fix reduces MoE's depth drop | drop(M-expert) − drop(M-stack) ≥ +2 | ≤ +0.5 |
| **H28** | The fix improves depth-16 accuracy | M-expert − M-stack at depth 16 ≥ +1.5 | — |
| **H29** | No cost when shallow | \|M-expert − M-stack\| at depth 2 ≤ 1 | — |
| **H30** (mechanism) | The init ratio explains the difference | transform/residual RMS ratio at init is ≥ 5× larger under M-expert | — |

## X2. Data-fitted vs fixed-random VQ codebook

**Defect.** With the repo config (`ema_update=False`, outer `learnable_codebook=False`, `orthogonal_reg_weight=1`), the codebook receives **no gradient from reconstruction** (measured: `None`) and only commitment/orthogonality gradient (measured norm 0.0076). The tokens are therefore a near-orthogonal random dictionary.

**Pipeline (STEM-faithful mini-pretraining on Cora, unsupervised):**

1. **Pretrain.** Encoder (SAGE, 2 layers, hidden 128, BatchNorm, dropout 0.15) → `VectorQuantize(dim=128, codebook_size=128, codebook_dim=128, heads=4, separate_codebook_per_head=True, decay=0.8, commitment_weight=0.25, use_cosine_sim=True, orthogonal_reg_weight=1, orthogonal_reg_max_codes=32, kmeans_init=False)`.
   - Losses, matching `pt_model.py`: feature reconstruction (masked features, `feat_p=0.2`, linear decoder from the quantized output) + topology reconstruction (inner-product decoder, negative sampling) + VQ loss.
   - Augmentation: `mask_feature` and `dropout_adj(p=0.2)`.
   - AdamW, lr 1e-3, 200 epochs, full batch.
2. **Probe.** Freeze the encoder and VQ. Train a linear classifier on the concatenated codes (the 4th VQ output, as `ft_model` uses) with training labels. Select on validation and report test.

**Variants:**

| ID | Codebook |
|---|---|
| **C-fixed** | repo config (`ema_update=False`) |
| **C-ema** | `ema_update=True`, everything else identical |
| **C-none** [reference] | the same pretraining with no quantization (probe on z) |

**Diagnostics:** code-usage perplexity per head; fraction of dead codes; reconstruction losses at the end of pretraining.

**Hypotheses (paired):**

| ID | Hypothesis | Support if | Refuted if |
|---|---|---|---|
| **H31** | A data-fitted codebook gives better tokens | C-ema − C-fixed probe accuracy ≥ +2 | ≤ 0 |
| **H32** | It uses more of the codebook | C-ema code perplexity > C-fixed | — |
| **H33** | Context only: the size of quantization's information loss | C-none − best(C-*) | — |

## Threats to validity

- One dataset.
- Width 128, not 768.
- Single-dataset pretraining, not STEM's multi-domain pretraining.
- Linear probe rather than STEM's full finetuning.
- MoE nondeterminism.
- X2 uses the SAGE encoder only (MoE is studied in X1).
