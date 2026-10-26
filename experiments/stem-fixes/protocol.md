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

## Amendment 1 (2026-10-26, from the Codex design review, BEFORE any full run)

Only the 1-seed smoke test had been run. These amendments supersede the matching parts above.

**Corrected defect statement (X2).** With outer `learnable_codebook=False`, *neither* reconstruction *nor* commitment updates the codebook. Commitment trains the encoder and `project_in` path. The codebook's only gradient is the orthogonality loss, plus AdamW weight decay.

**X1 pairing.** Both arms build identical encoder, router, BatchNorm and head weights from one initialization seed. M-expert re-initializes only the expert tensors, inside `torch.random.fork_rng`. All RNGs (Python, NumPy, Torch) are then reset to the same training seed for both arms.

**X1 hypotheses (all in percentage points).** Define depth_drop(v) = acc(v, depth 2) − acc(v, depth 16), so positive means a loss with depth. Define depth_benefit = depth_drop(M-stack) − depth_drop(M-expert).

| ID | Role | Support | Refute | Otherwise |
|---|---|---|---|---|
| **H27 (PRIMARY)** | depth_benefit | ≥ +2 and the paired 95% CI excludes 0 | ≤ +0.5 | inconclusive |
| **H28 (secondary)** | M-expert − M-stack at depth 16 | ≥ +1.5 | — | — |
| **H29 (secondary, non-inferiority)** | M-expert − M-stack at depth 2 | ≥ −1, i.e. no shallow loss larger than 1 point | — | — |
| **H30 (manipulation check)** | mean per-layer mixed transform/residual ratio at init | ≥ 5× larger under M-expert | — | — |

H30 is a manipulation check, not evidence that the ratio explains accuracy. The X1 diagnostics now include per-layer *individual* and *router-mixed* ratios, captured with forward pre-hooks, plus both entropy of mean routing and mean per-node routing entropy.

**X2 renamed: "STEM-component VQ pilot"** (feature + topology reconstruction with the real Encoder and VQ). It is not a faithful miniature of STEM pretraining: the repo uses loss weights 100 / 0.01 / 10, a semantic teacher, edge reconstruction, lr 1e-4, gradient clipping and a scheduler. Conclusions are limited to this objective.

**X2 arms (replacing C-fixed and C-ema).** All keep outer `learnable_codebook=False`:

| Arm | `ema_update` | `orthogonal_reg_weight` | Role |
|---|---|---|---|
| C-repo | False | 1 | repo reference |
| C-random | False | 0 | fixed random codebook (a buffer) |
| C-ema | True | 0 | EMA-fitted codebook |
| C-none | — | — | no VQ |

**X2 hypotheses:**

| ID | Role | Support | Refute | Otherwise |
|---|---|---|---|---|
| **H31 (PRIMARY)** | C-ema − C-random code-probe accuracy | ≥ +2 points and the paired CI excludes 0 | ≤ 0 | inconclusive |
| **H32 (secondary diagnostic)** | perplexity(C-ema) > perplexity(C-random), per head | as stated | — | — |
| **H33 (renamed)** | continuous-reference gap, C-none − each C-* arm | descriptive only; not "information loss" | — | — |

**X2 diagnostics and pairing:**

- Probes on frozen `z` from every pretrained encoder, to separate encoder quality from token quality.
- 3 probe seeds per frozen representation (mean reported), separate from the pretraining seed.
- Decoders are built *before* the VQ, so shared modules initialize identically across arms.
- Augmentation and negative sampling are re-seeded every epoch from (pretraining seed, epoch), so every arm sees the same corruptions.
- Final reconstruction diagnostics are computed in eval mode on clean inputs with fixed negatives.
- Per-head perplexity and dead-code fraction.

Decay stays at 0.8 (repo config); no tuning on test.

**Reporting:**

- Every paired contrast, means by split, and the overall mean.
- Uncertainty is labeled as conditional on this graph and these splits.
- Run files are unique per launch and record the package versions.
- Splits {0, 1, 2}.
- A fresh-process repeatability check is run before launch.

**Not a general STEM fix without confirmation:** a larger width and another codebook size would be pre-specified follow-ups.

## Threats to validity

- One dataset.
- Width 128, not 768.
- Single-dataset pretraining, not STEM's multi-domain pretraining.
- Linear probe rather than STEM's full finetuning.
- MoE nondeterminism.
- X2 uses the SAGE encoder only (MoE is studied in X1).
