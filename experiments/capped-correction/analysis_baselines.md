# Analysis: capped correction versus baselines

Protocol: `baselines_protocol.md` (written and Codex-reviewed before the run). Code:
`STEM-GNN/scripts/capped/baselines.py`, `analyze_baselines.py`. Raw output: `baselines_output.txt` (run on
the cluster, Slurm 33083377). Runs: Slurm array 32902262, all 15 cells (Cora, splits 0–2 × seeds 0–4)
completed with no errors; one code hash. The selfcheck passed beforehand.

**Integrity check.** For every original arm, cell and budget, clean, certified and protocol-union worst-node
were reproduced exactly (max drift 0), and adding the baseline graphs never raised any arm's worst-node.
Every number below uses the extended union (all graphs from both runs). That lowers S0, RES and the larger
caps slightly compared with `analysis.md` (S0 34.8 → 31.0); CAP0.1 moves 68.3 → 68.2.

## Bottom line

- **CAP0.1 and CAP0.25 improve on the prespecified envelope of tested baseline settings, on average.**
  - Their mean per-cell worst-node@20% advantages are +9.62 and +8.96 points, significant after Holm
    correction over five tests (Family F).
  - Eligibility allows baselines up to 0.5 clean-accuracy points below CAP.
  - This establishes improvements at two tested CAP settings. It doesn't show superiority across a
    continuous frontier or over untested baseline settings.
- **At CAP0.5 (75.0 clean), no difference from the eligible baseline envelope is established:** +1.74
  points, 95% CI [−0.04, +3.52], Holm p = .0552.
- **At CAP0.75 and CAP0.9, the mean envelope gaps are significantly negative** (−4.7 and −10.6).
  - RUNG-lite is a strong high-clean baseline (81.7 clean, 49.9 worst-node). Its mean robustness is
    close to CAP0.75's, with 5.42 more points of clean accuracy.
  - The exploratory paired robustness difference, CAP0.75 minus RUNG-lite, is +0.35 [−7.51, +8.21].
    That does not establish that they are equivalent.
- **The graph-free MLP-M remains the most robust model** (70.0 / 70.0). CAP's segment sits just above it
  in clean accuracy: +1 to +3 clean for −1.8 to −5.4 worst-node.
- **The low-cap gain is partly a coverage gap.** No tested baseline setting has *mean* clean accuracy
  between 71 and 73. The envelope there is usually set by cleaner baselines (RUNG-lite, ENS0.1, TextGate)
  or by MLP-M. Blends with a weight below 0.1, calibrated blends or tuned robust baselines could change
  this; none was tested.

## Arm means (15 cells, points, extended union)

| Arm | Clean | Worst-node@20% | Worst-node@10% | Worst-graph@20% | SAGE transfer@20% | Certified |
|---|---|---|---|---|---|---|
| MLP-M | 70.0 | 70.0 | 70.0 | 70.0 | 70.0 | — |
| CAP0.1 = CAP\* | 71.0 | 68.2 | 68.6 | 69.6 | 70.5 | 65.9 |
| CAP0.25 | 72.7 | 64.6 | 66.0 | 68.7 | 71.1 | 58.8 |
| CAP0.5 | 75.0 | 56.8 | 59.7 | 65.7 | 71.0 | 43.1 |
| ENS0.1 | 75.3 | 51.3 | 55.6 | 62.6 | 69.6 | — |
| CAP0.75 | 76.2 | 50.2 | 54.9 | 63.9 | 70.8 | 23.0 |
| CAP0.9 | 76.5 | 44.3 | 50.4 | 61.7 | 69.3 | 9.7 |
| RES | 76.8 | 40.9 | 48.0 | 60.1 | 68.0 | — |
| ENS0.25 | 77.1 | 41.7 | 48.3 | 58.4 | 68.5 | — |
| S0 (STEM-GNN) | 78.3 | 31.0 | 40.1 | 54.9 | 65.8 | — |
| ENS0.5 / 0.75 / 0.9 | 78.3–78.5 | 31.3–34.9 | 40.3–43.2 | 55.1–56.1 | 66.0–67.4 | — |
| SAGE+TextGate (reference) | 79.4 | 48.7 | 57.0 | 59.5 | 77.5 | — |
| SAGE | 81.4 | 27.1 | 37.3 | 54.4 | 65.2 | — |
| GCN | 81.7 | 22.1 | 33.2 | 48.5 | 68.7 | — |
| RUNG-lite | 81.7 | 49.9 | 57.2 | 68.7 | 76.7 | — |
| APPNP-lite | 81.9 | 14.3 | 27.3 | 45.1 | 66.9 | — |

## Family F: matched-clean frontier (confirmatory, Holm over 5)

Per cell, D_q = CAP(q) worst-node@20% − the best worst-node among baselines whose clean accuracy is at
least CAP(q)'s minus 0.5 points. No cell lacked an eligible baseline.

| Arm | Mean D [unadjusted 95% CI] | Wins | Holm p | Verdict |
|---|---|---|---|---|
| CAP0.1 | +9.62 [+4.75, +14.48] | 11/15 | .0025 | Above the baseline envelope |
| CAP0.25 | +8.96 [+5.74, +12.18] | 13/15 | .0002 | Above the baseline envelope |
| CAP0.5 | +1.74 [−0.04, +3.52] | 9/15 | .055 | No difference shown |
| CAP0.75 | −4.67 [−7.26, −2.09] | 3/15 | .0034 | Below the baseline envelope |
| CAP0.9 | −10.59 [−14.42, −6.76] | 1/15 | .0002 | Below the baseline envelope |

Which baseline sets the envelope, per cell (counts over 15):
- At CAP0.1: RUNG-lite 4, ENS0.1 4, MLP-M 4, TextGate 3.
- At CAP0.25: RUNG-lite 5, ENS0.1 3, TextGate 3, ENS0.25 2, MLP-M 2.
- At CAP0.5: RUNG-lite 7, TextGate 3, ENS0.1 2, MLP-M 2, ENS0.25 1.

No tested baseline has mean clean accuracy between 71 and 73 points: MLP-M averages 70.0 and ENS0.1
averages 75.3. Individual cells do enter this range, including ENS0.1 in three cells and ENS0.25 in one.
The envelope counts describe a comparator that changes from cell to cell, not a single matched baseline.

## Family B: robustness of CAP\* (confirmatory, Holm over 10)

CAP\* (= CAP0.1 in every cell) beats each of the ten Family B comparators on worst-node@20% in 15/15
cells. MLP-M is excluded from Family B and remains more robust than CAP\*.
- RUNG-lite +18.3 [+14.6, +22.0]; SAGE+TextGate +19.5; ENS0.1 +16.9 [+9.7, +24.2].
- GCN +46.1; SAGE +41.1; APPNP-lite +53.9.
- ENS0.25 to ENS0.9: +26.4 to +36.9.
- All Holm p < .001.

But it is 4.3 to 10.9 points less clean than each of them. As the protocol says, this only shows that the
fixed selected arm is more robust, which is expected for a model close to MLP-M. It isn't a frontier claim.

## Descriptive

- **Pareto set of arm means:** MLP-M, CAP0.1, CAP0.25, CAP0.5, ENS0.1, CAP0.75, RUNG-lite, GCN, APPNP-lite.
  - CAP0.75 is on it only by 0.3 worst-node points over RUNG-lite. RUNG-lite is 5.4 points cleaner, and
    their paired worst-node difference is +0.35 [−7.5, +8.2].
  - GCN and APPNP-lite are on it only through tiny clean margins.
- **Selected comparisons with ENS settings of nearby mean clean accuracy (exploratory, uncorrected):**
  - CAP0.5 vs ENS0.1 (nearly equal clean, −0.35): +5.5 worst-node, CI [−0.9, +11.9].
  - CAP0.75 vs ENS0.25 (clean −0.8): +8.5 [+4.3, +12.7].
  - CAP0.9 vs ENS0.25 (clean −0.6): +2.6 [−2.1, +7.3].
  - These suggest a possible advantage over some tested blends. They don't establish that CAP beats
    blending at matched clean accuracy: two of the three robustness CIs include zero, ENS is
    uncalibrated, and CAP also trains its correction through the combined output.
  - ENS0.1 vs CAP0.75 and CAP0.9 goes the other way, and a finer ENS grid wasn't tested.
- **RES vs ENS0.25 (exploratory, uncorrected):** clean −0.23; worst-node −0.88 [−6.57, +4.81]. This
  establishes neither an advantage nor equivalence. The earlier confirmatory H82 supports CAP\* over RES,
  with lower clean accuracy; it does not establish CAP over blending.
- **Splits:** CAP\* − baseline worst-node has the same sign in all three splits for every baseline.

## Caveats

- **Grid gap.** No tested baseline setting has mean clean accuracy between 71 and 73 points. ENS weights
  below 0.1, calibrated blends or tuned robust baselines could change the envelope there; none was
  evaluated.
- **Attack strength.**
  - CAP's worst-node at larger caps is accuracy against the tested attacks only (certified accuracy is
    much lower: 43.1 at CAP0.5).
  - RUNG-lite is a simplified reimplementation evaluated without a RUNG-specific adaptive attack. Its
    measured 49.9 may overestimate its robustness, which makes the current comparison harder for CAP.
    Stronger attacks could change the high-clean comparison.
- **Baselines untuned.** They use the repo's existing settings. The envelope is a max over 11 noisy arms,
  which biases it upward (conservative for CAP).
- **Scope.** Cora only; one dataset with 3 splits × 5 seeds, so CIs describe split and training
  variability, not replication across datasets. Edges only; test-label PRBCD.
- **Dependence.** The paired t tests treat the 15 cell differences as independent. Shared graph structure
  and overlapping split information may make them dependent, so treat nominal CI coverage and p-values
  with caution. Holm addresses multiplicity, not this dependence.
- **No novelty claim** before a prior-art check on bounded or certified graph corrections.
