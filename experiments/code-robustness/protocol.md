# Protocol: code-level robustness for STEM-GNN (CAC, VBM, commitment)

Written 2026-10-28, before any result. Codex will review the design before any full run.

**Constraint (user):** no method uses text embeddings to weight or select graph structure. Every regularizer acts on encoder states and VQ codes. Perturbations are structure-only (DropEdge).

## Model (STEM-faithful components)

- **Encoder:** real `Encoder` (SAGE, 2 layers, hidden 128, BatchNorm, dropout 0.15).
- **VQ:** real `VectorQuantize(dim=128, codebook_size=128, codebook_dim=128, heads=4, separate_codebook_per_head=True, use_cosine_sim=True, ema_update=True, decay=0.8, orthogonal_reg_weight=0, commitment_weight=0.25)`. This is the X2 `C-ema` configuration, the best codebook in `../stem-fixes`.
- **Decoder:** as in `ft_model`, a linear map from the concatenated codes (512-d) to heads × classes, with logits averaged over heads.

## Pipeline per (split, seed)

1. **Pretrain once and share across methods.** Encoder + VQ with feature + topology reconstruction, as in the X2 pilot (200 epochs).
2. **Finetune** each method from the same pretrained state. The codebook is frozen (`freeze_codebook`, mirroring `--freeze_vq 1`). The encoder and decoder train with cross-entropy plus the method's regularizer. Adam lr 0.01, wd 5e-4, up to 300 epochs, patience 50 on validation accuracy.

## Methods (regularizer weight λ = 1 for all; no tuning)

| ID | Regularizer |
|---|---|
| **B0** | none (the repo: VQ loss discarded during finetuning) |
| **B1 Commitment** (existing; control) | mean ‖ẑ − c(ẑ)‖², where ẑ is the per-head L2-normalized `project_in(z)` and c is its assigned code |
| **CAC** (new) | per-head soft assignment p = softmax(cos(ẑ, C)/0.1). Loss = KL(p_clean.detach() ‖ p_view), with the view = 30% undirected DropEdge |
| **VBM** (new) | per head, the top-2 codes a, b give the normalized boundary distance m = (s_a − s_b)/‖c_a − c_b‖. Loss = mean relu(0.05 − m) |

## Threats (evasion; test nodes; budget = 10% and 20% of undirected edges)

| Attack | How it works |
|---|---|
| **STE-PRBCD** | white-box. During the attack only, a wrapper replaces hard codes with `ẑ + (c − ẑ).detach()`; accuracy is measured with hard VQ |
| **Transfer** | the PRBCD graph from a SAGE surrogate, trained on the same split and seed |
| **Worst** | per-node minimum correctness across the two attacks (a node counts as correct only if correct under both) |

## Hypotheses (paired by split and seed; n = 15; percentage points; paired t 95% CI)

| ID | Hypothesis | Support if | Refute if | Otherwise |
|---|---|---|---|---|
| **H34 (PRIMARY)** | CAC − B0 on worst@20% | ≥ +3, CI excludes 0, and clean loss ≤ 1 (clean CAC − B0 ≥ −1) | ≤ +0.5 | inconclusive |
| **H35** | VBM − B0 on worst@20% | ≥ +3, CI excludes 0, and clean ≥ −1 | ≤ +0.5 | inconclusive |
| **H36 (control)** | B1 − B0 on worst@20% | ≥ +3, CI excludes 0 | ≤ +0.5 | inconclusive |
| **H37 (novelty-relevant)** | CAC − B1 on worst@20% | ≥ +1.5 | ≤ 0 | — |
| **H38 (mechanism)** | code-change rate under STE-PRBCD@10% is lower for CAC than for B0 | as stated | — | — |

H37 asks whether CAC beats the existing mechanism.

## Diagnostics

- Clean accuracy.
- Accuracy under each attack separately.
- Per-head code-change rate under STE-PRBCD@10%.
- Number of realized edits per attack, asserted to be ≤ budget.

## Amendment 1 (2026-10-28, from the Codex design review, BEFORE the full run)

**Observed before amending.** A one-cell smoke run (split 0, seed 0, 10 pretraining epochs, 10% budget only, old attack suite) gave worst@10%:

| Method | worst@10% | Code change | Clean |
|---|---|---|---|
| B0 | 57.9 | 0.165 | 75.7 |
| B1 (strong) | 66.1 | — | 83.4 |
| CAC | 68.1 | 0.083 | — |
| VBM (old top-two) | 56.7 | — | — |

It is a debug run only and is excluded from all inference.

**Methods (final):**

| Method | Definition |
|---|---|
| **B0** | no regularizer |
| **B1n** | native commitment: `0.25 · F.mse_loss(ẑ, c)`, the real VQ loss strength |
| **B1s** | strong commitment: per-head squared distance, 512× native (exploratory) |
| **CAC** | Σ_k KL per head, **averaged over heads** (`.sum(-1).mean()`). Both clean and view assignments come from a **deterministic forward** (dropout off, BatchNorm on running statistics), with gradients through the view. The view's DropEdge uses a separate RNG stream seeded from (cell, epoch). |
| **PredCons** | **matched control:** identical views and deterministic forward, but the KL is on *class predictions* (temperature 1) instead of code assignments. It separates code-specific from ordinary consistency. |
| **VBM** | **all-competitor** normalized margin, m = min_{b≠a} (s_a − s_b)/‖c_a − c_b‖, hinge at 0.05. Duplicate codes are audited (minimum pairwise code distance is logged). |

The CE forward is re-seeded per epoch identically across methods.

**Attack suite** (every candidate graph evaluated on the **hard** model; label-informed, global, transductive evasion on test predictions; edits allowed anywhere):

- **Own-model attacks, each at 10% and 20%:**
  - (a) STE-PRBCD;
  - (b) a second STE-PRBCD restart with a different seed;
  - (c) soft-backward / hard-forward BPDA, with codes = soft + (hard − soft).detach() and soft = expected code at attack temperature 0.1.
- **Transfer graphs:** (d) from an independently trained SAGE surrogate; (e) from the paired B0 model's attacks (a), (b) and (c).
- **Worst-node@r:** per-node correctness intersected over {clean graph} ∪ all candidates at budgets ≤ r.
- **Worst-graph@r:** the minimum over single candidate graphs.
- **Graph validity:** every candidate is checked for valid indices, no self-loops, no duplicates, symmetry, and 0 ≤ realized edits ≤ budget. Additions and deletions are logged.

**Hypotheses (n = 15 cells: splits {0, 1, 2} × seeds {0..4}; paired t 95% CI, conditional on this graph).** "Gain" is in worst-node@20%. A method passes the clean constraint if its mean clean change vs B0 is ≥ −1.

| ID | Comparison | Role |
|---|---|---|
| **H34** | CAC − B0 | **PRIMARY** |
| H35 | VBM − B0 | secondary |
| H36 | B1n − B0 | secondary |
| H37 | CAC − PredCons | secondary (code-specificity) |
| H37b | CAC − B1n | secondary |
| H38 | code-change rate on *common* graphs (SAGE transfer and B0 transfer) | diagnostic |

**Outcome rule:**

- **Support:** gain ≥ +3, the CI excludes 0, and the clean constraint holds.
- **Evidence against:** the upper CI bound is < +1.5.
- **Otherwise:** inconclusive.

Secondary hypotheses are reported with Holm-adjusted p-values. No coefficient, temperature or method is selected using test attacks.

**Novelty is not established.** Graph consistency (GRAND) and prototype margins (LVQ) are prior principles. Only the specific code-assignment formulation for STEM is under test.

## Threats to validity

- Cora only. Pubmed would be a pre-specified follow-up if a hypothesis is supported.
- Pilot pretraining.
- Width 128.
- CPU-scale PRBCD, which can be weak. PRBCD overstated CTV by 21 points. Results are framed as worst-over-attacks, and a stronger-attack check would be needed before claiming robustness.
- Novelty of CAC and VBM is unverified; a prior-art check is required before any claim.
