# Protocol: VGAE neighbor weighting, follow-up (stronger attacks, RUNG-VG, noise-aware VGAE, Pubmed)

Written 2026-11-24, before any run. Revised after the Codex design review and fix-check (`../reviews/codex_vgae_followup_fix_check_2026-11-24.md`)
(`../reviews/codex_vgae_followup_design_review_2026-11-24.md`, NO-GO), still before any run. Code:
`STEM-GNN/scripts/vgae_weighting/followup.py`, analysis `analyze_followup.py`. It builds on `../vgae-weighting/`
(same τ = 0.3, ε = 0.2, VGAE, recipes and seeds).

**Why:** the first experiment found a +3.09-point adaptive gain for S-VG over S0 on Cora (H103). It left four
open questions, which the user asked to test:
1. Is the gain partly attack weakness?
2. Do VGAE weights help inside the strongest baseline, RUNG-lite?
3. Does a VGAE trained to resist noise fix the null random-injection result?
4. Does it replicate on a second dataset?

**Constraint, as applied here.** The *added* edge weights (VGAE, masked VGAE, pruned VGAE, Jaccard) use graph
structure only and never read text. Two qualifications:
- **RUNG-lite's own propagation coefficients depend on model predictions, which read the node text features.**
  So RUNG-lite and RUNG-VG are *not* structure-only defenses. RUNG-VG is run because the user asked for "VGAE
  weights inside RUNG-lite"; it is reported with this qualification.
- The STEM and SAGE arms have no prediction-dependent edge coefficients.

The attacker choosing edits through the model's loss is the threat model, not part of the defense.

## New arms and scorers

| Arm | What |
|---|---|
| RUNG-VG | RUNG-lite (λ 0.3, δ 0.3, γ 1.2, as before) with VGAE neighbor weights as its input edge weights. Each target's weights are rescaled to keep their total, so uniform scores give back RUNG-lite (up to numerical tolerance). It uses the same VGAE, the same τ/ε and the same fit seed as RUNG-lite (base + 702). The two fits start from the same initialization; their training paths can still differ. |
| S-VGm | STEM with a masked-edge VGAE (MaskGAE-style). Each epoch, the encoder sees a random 50% of undirected edges and reconstructs the hidden 50% against negative samples. Inference encodes the full graph. |
| S-VGp | STEM with a self-pruned VGAE. Score every edge with the standard VGAE, drop the lowest 30% of undirected edges, fit a fresh VGAE on the pruned graph, and fix its posterior means. An edge's score is the cosine of those fixed means. |

The 50% mask and 30% prune are fixed in advance, with no tuning.

The Jaccard score is computed from sparse two-hop counts, converted to float32 before the original degree scatter
and division. For the intended simple unweighted graphs, the selfcheck requires bit-for-bit equality with the
original on the tested Cora graphs. Pubmed peak memory and runtime are measured in the feasibility gate.

## Part A: adaptive evasion with a stronger, weight-aware attack suite

**Cells.**
- **Cora:** the 15 original cells (splits 0–2 × seeds 0–4). They reuse the original checkpoints: the same
  trained models (the eight shared arms), the surrogate and the 52 attacked graphs.
  - Each checkpoint's tag, Cora data hash and code hash are verified against the current files.
  - Every original metric must reproduce exactly on the original graphs.
- **Pubmed:** 9 cells (splits 0–2 × seeds 0–2, from Pubmed's own splits), trained from scratch with the same
  recipe and seed formula.
  - Pubmed's splits have 60 training nodes, 500 validation nodes and about 19,157 test nodes.

**Attacked arms:** S0, S-VG, SAGE, SAGE-VG, RUNG-lite and RUNG-VG. Budgets are 10% and 20% of the undirected
edges.

**Graphs per cell (per budget), 89:**
1. **Standard suite (31).** The original five PRBCD variants on each attacked arm, plus SAGE-surrogate transfer.
   On Cora this is reused; only RUNG-VG's graphs are new.
2. **GCN transfer (1).** A strong PRBCD on the independently trained GCN, transferred.
3. **Strong suite, every attacked arm (48 = 6 × 8).** Seven strong PRBCD runs (400 epochs, 300 with resampling,
   200,000-edge block):
   - learning rate ×1 (two seeds), ×0.1, and ×10 (two seeds), all `ste`/prob-margin;
   - `soft`/prob-margin and `ste`/tanh-margin at ×1;
   - plus a distinct greedy RBCD attack (GRBCD; 125 steps, prob-margin).

   This sweep runs through the *true* differentiable VGAE path for the weighted arms, and identically for
   the unweighted ones.
4. **Weight-aware attacks, weighted arms only (9 = 3 × 3).** S-VG, SAGE-VG and RUNG-VG; strong settings,
   prob-margin.
   - **bpda_flat:** the forward pass uses the true weights; the backward pass uses the derivative of the uniform
     weights w/Σw.
   - **detach:** no gradient flows through the scorer.
   - **twin:** the same trained model with uniform weights (τ → ∞, a deep copy) is attacked, and the graph is
     transferred.

   bpda_flat and detach both suppress the scorer gradient, so they test obstruction from the scorer, not a
   larger true gradient. The learning-rate sweep covers that.

Every graph is validated (edit count within budget, symmetric, no self-loops or duplicates) as soon as it is
produced, then saved atomically in its own file with its job key, configuration and the checkpoint's identity.
- **On resume,** each file must match all three and is validated again. Unexpected files, or graph files without a
  checkpoint, stop the run.
- The original Cora graphs are imported the same way, after the checkpoint identity is saved.

**Endpoint: worst-node accuracy over the expanded union.** This is all 89 graphs of a budget level and below,
plus the clean graph. Every arm is scored on it, including the unattacked S-SJ, GCN and MLP-M.

**Effort is not matched exactly.** Weighted arms get three extra targeted attacks. The comparison is fair *on the
shared union*, since every arm is scored on the same graphs, but it does not establish equal attack strength.

**Sensitivity diagnostic (descriptive).** The paired change in S-VG's own-attack worst-node@20% when the
weight-aware graphs are added to standard plus strong, including individual cells. A drop of at least 3 points
flags sensitivity to the added attacks; it does not identify the cause. No drop does not rule out gradient
masking.

## Part R: random edge injection (poisoning, as before)

- 5 levels × 10 graphs per dataset, using the original seed formulas.
- **Cora** uses 0/500/1000/2000/5000 added edges.
- **Pubmed** uses the same fractions of its undirected edge count: 0, 9.5%, 18.9%, 37.9% and 94.7%.
- Arms: S0, S-VG, S-VGm, S-VGp, S-SJ, SAGE, SAGE-VG, GCN, RUNG-lite, RUNG-VG and MLP-M.
- Edge AUC (original versus injected) for S-VG, S-VGm, S-VGp and S-SJ.
- **Reproduction:** on Cora, the eight arms shared with the original experiment must reproduce its accuracy and
  macro-F1 exactly in all 50 cells.

## Hypotheses (one Holm family; two-sided paired t; 95% CI; points)

**What the statistics mean.** The cell-level paired summaries, t intervals and Holm-adjusted p-values are
operational decision rules, conditional on these fixed graphs and splits. Cells are dependent:
- seeds share a graph and test population;
- only three splits are reused.

They are not independent dataset replications and support no population-level claim. All paired cell
differences and split-level means are reported.

- **Support:** mean ≥ +2 and Holm p < .05 against zero.
- **Evidence against:** upper CI < +1.

| ID | Comparison | Endpoint | n |
|---|---|---|---|
| H110 | S-VG − S0 | Cora A worst-node@20%, expanded union | 15 |
| H111 | RUNG-VG − RUNG-lite | Cora A worst-node@20%, expanded union | 15 |
| H112 | S-VGm − S0 | Cora R accuracy, 5000 added edges | 10 |
| H113 | S-VGp − S0 | Cora R accuracy, 5000 added edges | 10 |
| H114 | S-VG − S0 | Pubmed A worst-node@20%, expanded union | 9 |
| H115 | RUNG-VG − RUNG-lite | Pubmed A worst-node@20%, expanded union | 9 |
| H116 | S-VGm − S-VG | Cora R accuracy, 5000 added edges | 10 |
| H117 | S-VGp − S-VG | Cora R accuracy, 5000 added edges | 10 |

**Guardrails (not in the family).** The clean comparison of the same pair must have mean ≥ −1 and lower CI > −2:
- Part A clean accuracy for H110, H111, H114 and H115;
- Part R at 0 added edges for H112 and H113.

**Claims.**
- **C1:** S-VG retains a gain on Cora's declared expanded attack union (H110 plus its guardrail). This is
  evidence against the attacks in that suite, not evidence that gradient masking or stronger attacks are absent.
- **C2:** the same on Pubmed (H114 plus its guardrail).
- **C3:** VGAE weights add robustness inside RUNG-lite: H111 (Cora) and H115 (Pubmed), each with its guardrail,
  subject to the RUNG text qualification above.
- **C4:** a noise-aware variant improves accuracy over S0 under 5000 injected edges (H112 or H113, plus its
  guardrail) *and* over the original S-VG (H116 or H117 for the same variant).

**What would count as a new frontier.** C1 and C2 together are a *replication signal on two fixed graphs*, not
enough by themselves to establish a new frontier. A frontier would also need:
- independent datasets or graph instances;
- stronger, independently validated attacks;
- comparisons with competitive, tuned baselines.

The report will say which of these are still missing.

**Descriptive:**
- Pubmed Part R, including S-VG, S-VGm and S-VGp versus S0 at every level.
- Edge AUCs.
- Own-attack worst-node per arm for the standard, +strong and +weight-aware sets, and per-attack accuracy.
- Old-union versus expanded-union S-VG − S0.
- GCN and SAGE transfer.
- RUNG-VG versus MLP-M.
- Per-split effects.

## Feasibility gate (before the full grid)

**Before submitting the full grid:**
1. **Selfcheck.** Correctness tests on the real graphs with short attack settings. This is an integration check,
   not a tiny synthetic unit test:
   - hand-computed derivatives for bpda_flat and detach, including a zero edge weight, and restoration of the
     weight function by identity;
   - sparse Jaccard bit-identical to the dense version;
   - exact reproduction of every original metric from one old checkpoint, and the provenance checks;
   - twin isolation, and restoration of the attack mode and eval state;
   - every attack type returning a valid graph (short settings);
   - pickling in a fresh process.
2. **Smoke cells:**
   - one Cora Part A cell (split 0, seed 0), interrupted once during attacks and resumed;
   - one Cora Part R cell (level 5000, graph 0), checked for exact reproduction of the original JSON;
   - one Pubmed Part A cell and one Pubmed Part R cell (level 94.7%), measuring peak memory, time per attack
     type and arm, scoring time, and per-graph file size.

**Pubmed amendment rule.** If a Pubmed Part A cell can't finish within three chained 12-hour jobs, one
dataset-wide amendment will reduce the Pubmed strong suite. In order:
1. drop learning-rate ×1 seed b and ×10 seed b;
2. then cut strong epochs to 200/150.

It will be decided from timing logs only, before any accuracy is scored (scoring happens only when a cell
finishes all attacks). It will be recorded here and reported as an amended suite. The weight-aware attacks,
GRBCD and the standard suite are kept.

## Declared risks

- **The attacks are stronger checks, not a certificate.** A stronger attack may exist; no loss trajectories are
  recorded, only the final graphs and their accuracies.
- **Cora Part A reuses trained models**, so its new-attack results are conditional on those 15 fits.
- **RUNG-lite is a simplified reimplementation, and the baselines are untuned.** RUNG's coefficients read text
  (see the constraint section).
- **Pubmed has a large test set (about 19k nodes) and a 60-label training set.** Attack budgets are
  graph-global.
- **No novelty claim.** Masked-edge GAEs (MaskGAE), structural pruning and similarity reweighting are known.
