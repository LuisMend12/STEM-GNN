# Analysis: structure-only VGAE neighbor weighting for STEM-GNN

Protocol: `protocol.md` (written and Codex-reviewed before any run). Code:
`STEM-GNN/scripts/vgae_weighting/run.py`, `analyze.py`. Raw output: `analysis_output.txt` (run on the cluster,
Slurm 34080666). Codex results review: `../reviews/codex_vgae_weighting_results_review_2026-11-24.md` (all
findings applied).

**Runs.**
- Selfcheck 34068322: 24/24 checks passed.
- Part R: array 34068530 plus smoke cell 34068323, 50 cells.
- Part A: array 34068531 plus smoke cell 34068324, 15 cells.
- All 65 cells completed with no errors, under one configuration (`3405b67aa1`).

The defense never reads text: the selfcheck confirmed that the edge weights passed to STEM are identical
when the text features change.

**Base graph.** OFA Cora has 5,278 undirected edges, so 500/1000/2000/5000 injected edges are
9.5/18.9/37.9/94.7%.

## Bottom line

- **Random edge injection (poisoning): the protocol claim is not met.**
  - S-VG − S0 is +0.03 at 2000 edges (H100: evidence against, under the operational rule) and +0.51 at
    5000 (H101: inconclusive).
  - S-VG − untrained VGAE is −0.00 at 5000 edges (H102: inconclusive).
  - SAGE-VG − SAGE is +0.29 at 5000 edges (H104: evidence against, under the operational rule).
- **Adaptive edge attacks (evasion): the prespecified STEM comparison meets its operational claim.**
  - S-VG − S0 worst-node accuracy over the tested attack union at 20% is +3.09 [+1.47, +4.70] points,
    13/15 cell wins, Holm p = .005 (H103 supported).
  - Clean accuracy changes by −0.34 [−1.65, +0.97] points. That meets the prespecified guardrail, but
    doesn't establish zero clean cost.
  - The analogous SAGE-VG − SAGE difference is descriptive: +4.30 [+3.77, +4.83] points on the tested
    union, with a rounded clean difference of +0.00 [−0.23, +0.24].
  - Attack-strength and gradient-masking concerns limit the robustness interpretation (see Caveats).
- **A post hoc hypothesis, not a tested mechanism:**
  - In Part R, the VGAE is fitted on each corrupted graph. Its mean original-versus-injected edge AUC
    ranges from 0.750 at 500 edges to 0.667 at 5000, below structural Jaccard at every level.
  - In Part A, the VGAE is fitted on the clean graph before the adaptive evasion attacks.
  - These observations motivate testing whether clean versus corrupted VGAE training matters. But the two
    parts also differ in attack type, model training and evaluation, and the AUC levels use different
    graphs and seeds. So neither the mechanism nor a causal noise effect is established.
- **S-VG's measured gain is relative to S0 on this attack suite.** RUNG-lite and MLP-M score higher in
  this setup (S-VG − RUNG-lite worst-node@20%: −14.57; MLP-M: 70.0).

## Confirmatory results (one Holm family; two-sided paired t; unadjusted 95% CI; points)

| ID | Comparison | Endpoint | n | Mean [95% CI] | Wins | Holm p | Verdict |
|---|---|---|---|---|---|---|---|
| H100 | S-VG − S0 | R accuracy, 2000 edges | 10 | +0.03 [−0.81, +0.86] | 6/10 | 1.000 | Against |
| H101 | S-VG − S0 | R accuracy, 5000 edges | 10 | +0.51 [−0.58, +1.59] | 7/10 | .954 | Inconclusive |
| H102 | S-VG − S-VGr | R accuracy, 5000 edges | 10 | −0.00 [−1.60, +1.59] | 5/10 | 1.000 | Inconclusive |
| H103 | S-VG − S0 | A worst-node@20% | 15 | +3.09 [+1.47, +4.70] | 13/15 | .005 | Supported |
| H104 | SAGE-VG − SAGE | R accuracy, 5000 edges | 10 | +0.29 [−0.02, +0.61] | 8/10 | .245 | Against |

Clean guardrails, both met:
- Part R at 0 edges: −0.18 [−1.82, +1.46].
- Part A clean: −0.34 [−1.65, +0.97].

These operational decisions are conditional on Cora:
- Cells share one graph; Part A's 15 cells share three splits, and Part R uses splits 4/3/3. The
  intervals describe run-level variation, not independent-dataset replication.
- The +2-point support threshold applies to the estimate; the significance test is against zero, not
  against +2.

## Part R: random edge injection (test accuracy, mean over 10 graphs, points)

| Arm | 0 | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|---|
| S0 | 77.66 | 77.20 | 76.37 | 74.07 | 71.21 |
| S-VG | 77.49 | 76.73 | 75.17 | 74.10 | 71.72 |
| S-VGr (untrained VGAE) | 77.59 | 76.32 | 75.63 | 74.37 | 71.72 |
| S-SJ (structural Jaccard) | 78.22 | 76.44 | 76.26 | 74.00 | 71.32 |
| SAGE | 81.33 | 80.15 | 79.68 | 78.40 | 76.65 |
| SAGE-VG | 81.31 | 80.05 | 79.62 | 78.45 | 76.95 |
| GCN | 81.81 | 80.67 | 79.26 | 77.26 | 73.06 |
| RUNG-lite | 80.84 | 81.09 | 80.32 | 79.40 | 77.97 |
| MLP-M | 70.03 | 69.92 | 69.20 | 69.64 | 69.90 |

Macro-F1 results are in `analysis_output.txt` and are descriptive. Each noise level uses its own corruption
and pretraining seeds, so comparisons across levels are descriptive; paired arm comparisons are within a
level.

**Edge AUC, original versus injected edges** (0.5 = no separation; milestone question 2; descriptive):

| Scorer | 500 | 1000 | 2000 | 5000 |
|---|---|---|---|---|
| Trained VGAE (S-VG) | 0.750 | 0.735 | 0.707 | 0.667 |
| Untrained VGAE (S-VGr) | 0.592 | 0.594 | 0.598 | 0.603 |
| Structural Jaccard (S-SJ) | 0.764 | 0.763 | 0.763 | 0.763 |

The trained VGAE scores injected edges lower than original ones on average, but less so at higher noise
levels, and a non-learned structural score separates at least as well at every level. Neither separation
turned into a supported accuracy gain under random injection.

## Part A: adaptive attacks (points, mean over 15 cells)

| Arm | Clean | Worst-node@10% | Worst-node@20% | Own-attack worst-node@20% | SAGE transfer@20% |
|---|---|---|---|---|---|
| S0 | 78.1 | 47.9 | 38.3 | 46.6 | 65.5 |
| S-VG | 77.8 | 51.0 | 41.4 | 53.8 | 69.1 |
| S-SJ (not attacked) | 77.4 | 49.0 | 39.2 | — | 64.8 |
| S-VGr (not attacked) | 77.8 | 48.9 | 39.0 | — | 65.6 |
| SAGE | 81.2 | 46.4 | 36.1 | 46.8 | 65.1 |
| SAGE-VG | 81.2 | 50.8 | 40.4 | 54.1 | 71.5 |
| GCN (not attacked) | 81.7 | 46.3 | 35.8 | — | 68.7 |
| RUNG-lite | 81.2 | 62.3 | 56.0 | 65.2 | 76.5 |
| MLP-M | 70.0 | 70.0 | 70.0 | 70.0 | 70.0 |

Unattacked arms have no own-targeted attack: the script's own-attack calculation contains only the clean
graph, so it returns clean accuracy. The table shows "—" rather than treating that value as an attack
result. MLP-M is also unattacked; it scores 70.0 on every graph because it ignores edges.

**Descriptive (uncorrected):**
- **S-VG − S-SJ worst-node@20%** on the shared tested union: +2.21 [+1.12, +3.30], 12/15 cell wins.
  - S-SJ was not directly attacked. Adding an S-SJ-targeted graph could only keep or lower its score on
    an expanded union.
  - But this comparison doesn't establish S-SJ's robustness against a tailored attack, or resolve
    differences in attack strength between methods.
- **SAGE-VG − SAGE:** +4.30 worst-node@20%, 15/15 cells; clean +0.00 [−0.23, +0.24].
- **Per split,** S-VG − S0 worst-node@20% is +2.30 / +2.55 / +4.40.

## Caveats

- **The adaptive result is limited to the tested attack union.**
  - Own-attack worst-node is 53.8 for S-VG versus 46.6 for S0, but those figures come from different,
    model-targeted attack sets.
  - The gap raises concern that PRBCD may optimize less effectively against the weighted model. It doesn't
    by itself prove gradient masking.
- **Transfer checks help but aren't conclusive.** The shared union includes attacks targeted at the
  unweighted S0, and SAGE-transfer accuracy is 69.1 for S-VG versus 65.5 for S0. Neither rules out a
  stronger attack tailored to S-VG.
- **More attack validation is needed** before reading the gain as reliable adaptive robustness: alternative
  gradients such as BPDA, and stronger optimization or more restarts. No certificate was produced.
- **Evasion versus poisoning.** In Part A, the VGAE and models are trained on the clean graph. The gain isn't
  shown for poisoning.
- **τ = 0.3 and ε = 0.2 were fixed, not tuned.** Other settings weren't tested; tuning them on a
  text-reading model would violate the constraint.
- **Single dataset and setup.** Cora (OFA, MiniLM) with the repo's 3 splits, not the Planetoid public split.
  Baselines are untuned, and RUNG-lite is a simplified reimplementation.
- **No novelty claim.** Similarity-based reweighting (GNNGuard, Jaccard) and VGAE edge scores are known. A
  prior-art check on *structure-only* latent edge weighting for robustness is needed before any claim.

## Is this a new frontier?

These results identify a candidate direction for further study: structure-only edge weighting fitted on a
clean graph.
- On this Cora setup, the prespecified STEM comparison meets its operational adaptive-attack claim. The
  GraphSAGE result is descriptive.
- The random-injection claim is not met, and the gains on the tested suite don't establish a new
  robustness frontier.

What would strengthen it, none run yet:
1. A stronger attack tailored to S-VG (BPDA on the weights, more restarts), to rule out gradient masking.
2. A directly attacked Jaccard control.
3. A direct test of clean- versus corrupted-graph VGAE training (for example a VGAE trained with edge
   dropout, or on a trusted subgraph).
4. VGAE weights inside RUNG-lite's robust aggregation.
5. A second dataset.
