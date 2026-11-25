# Protocol: structure-only VGAE neighbor weighting for STEM-GNN

Written 2026-11-24, before any run; revised after the Codex design review
(`../reviews/codex_vgae_weighting_design_review_2026-11-24.md`), still before any run. Code: `STEM-GNN/scripts/vgae_weighting/run.py`, analysis `analyze.py`.
Source idea: the user's course milestone, "Variational Autoencoder-Guided Neighbor Weighting for Robust
GCNs".

**Adapted under the standing constraint: the defense never uses text to weight, select, add or drop
edges.** The user chose the structure-only variant: the VGAE sees the graph only, never the MiniLM text
features. The edge-attack threat model (an attacker choosing edits through the model's loss) is unchanged
from earlier experiments.

## Method

1. **Structure-only VGAE.**
   - Encoder: a learned per-node embedding (64-d) → GCN (64) → GCN heads for μ and log σ (32-d latent).
   - Decoder: inner product σ(z_iᵀz_j).
   - Training: maximize the ELBO, i.e. edge reconstruction with negative sampling plus KL/N to N(0, I),
     for 200 epochs (Adam, lr 0.01). It trains on the *observed* graph, which in Part R is the corrupted
     one. Then it is frozen.
2. **Edge score:** s_vu = cos(μ_v, μ_u), recomputed on whatever graph the model is given. Under attack,
   the attacker's edges get scored too, and PRBCD differentiates through the VGAE encoder.
3. **Neighbor weights (the milestone's formula):**
   a_vu = (1 − ε)·softmax_{u∈N(v)}(s_vu / τ) + ε / |N(v)|.
   PRBCD's continuous edge weights multiply both terms. The weights sum to 1 over each node's neighbors.
4. **Used in:**
   - STEM-GNN's MoE weighted-mean aggregation (S-VG).
   - A plain weighted GraphSAGE, the milestone's update h_v = W_self h_v + Σ a_vu W_nbr h_u (SAGE-VG).
5. **τ = 0.3 and ε = 0.2, fixed before any run.** No validation search, for two reasons:
   - no text-reading model's validation accuracy picks the weights;
   - every arm gets one fit, so the weighted arms get no extra tries over S0.

## Arms

| Arm | What |
|---|---|
| S0 | STEM-GNN (Stage 2 pretrain and finetune recipe, unweighted) |
| S-VG | S0's pretrained encoder, finetuned with VGAE weights |
| S-VGr | Same, but the VGAE is untrained (random init): tests whether the *learned* structure matters |
| S-SJ | Same, but s = structural Jaccard of the endpoints' neighbor sets: a simple, non-learned structural score (descriptive comparison only) |
| SAGE | Plain weighted-mean GraphSAGE (hidden 64) |
| SAGE-VG | SAGE with VGAE weights (the milestone's model on a plain backbone) |
| GCN, RUNG-lite, MLP-M | References: standard GCN, the repo's robust propagation, graph-free text MLP |

## Part R: random edge injection (milestone / VGCN protocol, poisoning)

- Add 0, 500, 1000, 2000 or 5000 random undirected edges between distinct, unconnected nodes, with no
  duplicates and no self-loops.
- 10 corrupted graphs per level (graph g uses split g mod 3). Corruption seed = 50 000 + 1000·g + level.
- **Poisoning:** STEM pretraining, VGAE training, finetuning and every baseline all run on the corrupted
  graph, and evaluation is on it too.
- **Metrics:** test accuracy and macro-F1. For milestone question 2, the edge-score AUC (original versus
  injected edges) for the VGAE, the random VGAE and Jaccard. The injected-edge indicator is used only
  for this analysis, never as a model input.

## Part A: adaptive evasion (as in `../capped-correction`)

- 15 cells (splits 0–2 × seeds 0–4) on the clean graph.
- Attacked arms: S0, S-VG, SAGE, SAGE-VG and RUNG-lite. S-SJ is not attacked: its dense Jaccard is too
  costly inside PRBCD's 100 000-pair candidate blocks. Each gets five test-label PRBCD attacks
  at 10% and 20%, plus SAGE-surrogate transfer.
- Worst-node accuracy over the shared union of every graph built in the cell, plus clean.
- Every arm is scored on the union, including S-VGr, S-SJ, GCN and MLP-M, which aren't attacked
  themselves.
- **What counts as text-free:** the attacker's own choice of edges (using the model's loss) is the threat
  model, not part of our method. The constraint applies to the defense's edge weights, which never read
  text.

## Hypotheses (one Holm family; two-sided paired t; 95% CI; points)

These are **operational decision rules, conditional on Cora**:
- "Support" needs a point estimate ≥ +2 and a Holm-adjusted p < .05 against zero. It is not a test that
  the true effect is at least 2.
- "Evidence against" uses an unadjusted CI and is not a formal equivalence test.
- Cells share one graph, and Part A's 15 cells share 3 splits; Part R's 10 graphs use splits 4/3/3. So
  the intervals are run-level, not independent-dataset replications. Per-split effects are reported.

| ID | Comparison | Endpoint | n | Support | Evidence against |
|---|---|---|---|---|---|
| H100 | S-VG − S0 | Part R accuracy, 2000 added edges | 10 | ≥ +2, Holm p < .05 | Upper CI < +1 |
| H101 | S-VG − S0 | Part R accuracy, 5000 added edges | 10 | ≥ +2, Holm p < .05 | Upper CI < +1 |
| H102 | S-VG − S-VGr | Part R accuracy, 5000 added edges | 10 | ≥ +2, Holm p < .05 | Upper CI < +1 |
| H103 | S-VG − S0 | Part A worst-node@20% | 15 | ≥ +2, Holm p < .05 | Upper CI < +1 |
| H104 | SAGE-VG − SAGE | Part R accuracy, 5000 added edges | 10 | ≥ +2, Holm p < .05 | Upper CI < +1 |

**Clean guardrail (not in the family):** S-VG − S0 at 0 added edges (Part R) and on Part A clean accuracy
must each have mean ≥ −1 and lower CI > −2.

**Claims:**
- "Structure-only VGAE weighting makes STEM-GNN more robust to random edge injection" needs (H100 or
  H101), plus H102 and the Part R guardrail.
- "…to adaptive edge attacks" needs H103 and the Part A guardrail.
- H104 reports the milestone's method on a plain GraphSAGE.

**Descriptive:**
- Accuracy and macro-F1 curves over noise level for every arm.
- Edge AUC per level and scorer.
- S-SJ versus S-VG; chosen τ and ε.
- Part A clean, worst-node@10%, own-attack and transfer numbers, and the comparison with RUNG-lite and
  MLP-M.
- Per-split effects.

## Declared risks

- **The VGAE learns from the corrupted graph** (as the milestone notes), so it may learn to reproduce the
  injected edges. The edge AUC measures whether it does.
- **Adaptive attackers can target the scorer.** Under PRBCD, attacker edges are scored by the same frozen
  VGAE, and gradients flow through it. Gains against random noise may not carry over (H103 tests this).
- **Shared poisoning protocol, not seed-matched across levels.** Each level uses its own corruption and
  pretraining seeds, so compare arms within a level, not levels against each other.
- **Injection counts follow the milestone, but the base graph differs.** OFA Cora's undirected edge count
  is logged, and every level is also reported as a percentage of it, so it can be compared with
  Planetoid's 5,278.
- Cora (OFA, MiniLM) only; the repo's 3 splits rather than the Planetoid public split; baselines untuned;
  RUNG-lite is a simplified reimplementation. Part R uses 10 graphs per level, which only partly cross
  the 3 splits.
- No novelty claim: similarity-based neighbor reweighting (GNNGuard, Jaccard) and VGAE-based edge scores
  are known. The structure-only, text-free version inside STEM-GNN is what is tested.
