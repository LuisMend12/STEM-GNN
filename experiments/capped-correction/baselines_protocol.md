# Protocol addendum: baselines for the capped correction

Written 2026-11-06, after the capped run (`protocol.md`, `analysis.md`) finished and before any baseline run.
Revised after Codex review (`../reviews/codex_capped_results_review_2026-11-06.md`), still before any run.
Code: `STEM-GNN/scripts/capped/baselines.py`, analysis `analyze_baselines.py`.

## Why

The capped run compared CAP only against STEM's own variants (S0, RES) and the graph-free MLP-M. That
can't tell us whether the clean-versus-robust trade-off CAP traces beats what standard GNNs, the repo's
robust-propagation baselines, or a plain S0/MLP-M ensemble already give.

## Setting (unchanged)

- Same Cora splits 0–2 × seeds 0–4, same MiniLM features, same test-label PRBCD suite: five own attacks
  at 10% and 20%, plus SAGE-surrogate transfer.
- Each cell loads the capped run's checkpoint: the same trained S0, RES, CAP arms, MLP-M, surrogate and
  every attacked graph. Nothing from the capped run is retrained.

## Baselines

| Arm | What it is | Why |
|---|---|---|
| MLP-M | Graph-free MLP (from the capped run) | Immune to edges |
| GCN | 2-layer GCN, hidden 64 | Standard GNN |
| SAGE | 2-layer weighted-mean GraphSAGE, hidden 64 | STEM's backbone family without pretraining or VQ |
| APPNP-lite | Personalized-propagation smoothing (RUNGLite, `l2`) | Propagation baseline |
| RUNG-lite | Simplified RUNG robust IRLS propagation | Robust-aggregation baseline (repo version, not the reference code) |
| SAGE+TextGate | GNNGuard-style: edges weighted by frozen text cosine, τ = 10% edge-cosine quantile | Standard defense. Reference only: it lets text weight edges, which our method never does |
| ENS(a), a ∈ {0.1, 0.25, 0.5, 0.75, 0.9} | (1−a)·log p_MLP-M + a·log p_S0, no training | Untrained control built from the same two components CAP uses |

Hyperparameters are the repo's existing `adaptive_robustness` settings (no tuning here); training uses the
shared `fx.fit` recipe with validation early stopping.

About ENS: for predictions, log-probability mixing equals weighted logit mixing, so its effective balance
depends on the two models' logit scales. No temperature calibration is done. Unlike CAP, ENS doesn't train
a correction through the combined output. RES remains the closer control for the effect of the cap
itself; ENS answers "does CAP beat simply blending the two models?" ENS gets all five attacks, with S0's
STE and soft modes.

## Evaluation

- **Extended union.** Worst-node is recomputed for every arm, ours and the baselines, over the union of
  every graph built in the cell: the capped run's graphs plus every baseline's graphs. Our arms'
  worst-node can only stay the same or drop relative to `analysis.md`.
- **Reproduction is enforced, not just reported.** For every cell, our arms' clean accuracy, certified
  accuracy and protocol-union worst-node (both budgets) must equal the saved capped results to 1e-9, or
  the cell fails. Extended-union worst-node ≤ protocol-union worst-node is asserted per arm. The analysis
  asserts the same reproduction again.
- **Provenance.** Resume requires the same split, seed, rates, ENS weights, code hash, source-checkpoint
  SHA-256, data SHA-256, threads and library versions.

## Comparisons

**Family F: frontier (confirmatory, one Holm family over five tests).** For each CAP(q), q ∈ {0.1, 0.25,
0.5, 0.75, 0.9}, per cell:
- D_q = worst-node@20%(CAP(q)) − max worst-node@20% over the baselines whose clean accuracy is at least
  CAP(q)'s minus 0.5 points (all 11 baselines, including TextGate and the ENS weights).
- Cells with no eligible baseline are dropped and counted.
- Two-sided paired t on D_q.
- "CAP(q) improves on the baseline frontier at its clean level" if mean D_q > 0 with Holm p < .05;
  "below the frontier" if mean D_q < 0 with Holm p < .05.

Taking a max over 11 noisy baselines biases the envelope upward, so this test is conservative for CAP.
A positive result at one q means one CAP point improves the envelope at that clean level, not that CAP is
better across the frontier. The latter needs every q to be positive.

**Family B (confirmatory, separate Holm family, ten tests).**
- CAP\* − X on worst-node@20% (extended union), for X in GCN, SAGE, APPNP-lite, RUNG-lite, SAGE+TextGate
  and the five ENS weights.
- CAP\* is the arm chosen by the original selection rule, not re-chosen.
- This only answers whether the fixed selected arm is more robust than each baseline. CAP\* sits near
  MLP-M, so winning here can simply reflect less graph influence. It isn't a frontier claim.
- Clean CAP\* − X is reported beside it, descriptively.

**Descriptive and exploratory:**
- Pareto set of arm means on (clean, worst-node@20%).
- Pairwise CAP(q)/RES versus each baseline: Δclean and Δworst-node with uncorrected CIs, labelled
  "dominates" when both lower bounds are > 0.
- Per-split CAP\* − baseline worst-node.

These pairwise searches are not evidence for any claim.

## Risks

- The baselines aren't tuned for robustness. RUNG-lite is a simplified reimplementation, and the attacks
  are PRBCD with test labels only (no RUNG-specific adaptive attack), so their worst-node may be
  optimistic. That makes Family F harder for CAP, not easier.
- The extended union lowers every arm's worst-node, ours included, so compare only within this file, not
  against `analysis.md`.
- The 15 cells share Cora (3 splits × 5 seeds). CIs describe training and split variability on one
  dataset, not replication across datasets.
