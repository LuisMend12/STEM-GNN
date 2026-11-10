The numerical analysis reproduces, but several conclusions need tightening. **No BLOCKER found in the reported calculations.** I reran `analyze_baselines.py`: its entire output matches `baselines_output.txt`, excluding the shell’s exit-status line. I found no code bug in the inspected scoring or analysis that changes a reported number.

All arm-table values, Family F/B statistics, quoted pairwise differences, Pareto membership, and split-direction statements match. The extended-union shifts also check: S0 **34.8001 → 30.9736**, CAP0.1 **68.2818 → 68.1721**. Family F correctly uses Holm over five tests; Family B correctly uses a separate Holm family over ten.

The envelope counts supplied in your prompt are **exactly correct**, with no ties:

| CAP | RUNG-lite | ENS0.1 | ENS0.25 | MLP-M | SAGE+TextGate |
|---|---:|---:|---:|---:|---:|
| 0.1 | 4 | 4 | 0 | 4 | 3 |
| 0.25 | 5 | 3 | 2 | 2 | 3 |
| 0.5 | 7 | 2 | 1 | 2 | 3 |

Findings below refer to `experiments/capped-correction/analysis_baselines.md`.

- **SHOULD-FIX — lines 15–17: the headline overstates the unit of inference.** Family F establishes a positive *mean per-cell envelope gap*, with 11/15 and 13/15 wins. It does not establish superiority in every cell, an interpolated frontier segment, or superiority over untested baseline settings. “At least as clean” also omits the 0.5-point tolerance.

  Replace with:
  > **CAP0.1 and CAP0.25 improve the prespecified envelope of tested baseline settings on average.** Their mean per-cell worst-node@20% advantages are +9.62 and +8.96 points, respectively, significant after Holm correction over five tests. Eligibility allows baselines up to 0.5 clean-accuracy points below CAP. These results establish improvements at two tested CAP settings, not superiority across a continuous frontier or over untested baseline settings.

- **SHOULD-FIX — line 18: “tie” is not the protocol verdict.** Nonsignificance does not establish equivalence. Rounding the lower bound to “−0.0” further obscures this.

  Replace with:
  > **At CAP0.5 (75.0 clean), no difference from the eligible baseline envelope is established:** +1.74 points, 95% CI [−0.04, +3.52], Holm p = .0552.

- **SHOULD-FIX — lines 24–26, 67–68, 99–101: distinguish the gap in arm means from individual-cell coverage.** The statement that no baseline sits at 71–73 except MLP-M is false literally per cell: ENS0.1 does so in three cells and ENS0.25 in one. The envelope counts themselves are correct.

  Replace lines 67–68 with:
  > No tested baseline has mean clean accuracy between 71 and 73 points; MLP-M averages 70.0 and ENS0.1 averages 75.3. Individual-cell accuracies do enter this range, including ENS0.1 in three cells and ENS0.25 in one. The envelope counts describe a changing per-cell comparator, not a single matched baseline.

  Replace lines 99–101 with:
  > **Grid gap.** No tested baseline setting has mean clean accuracy between 71 and 73 points. ENS weights below 0.1, calibrated blends, or tuned robust baselines could change the envelope there; none was evaluated.

- **SHOULD-FIX — line 72: “every baseline” wrongly includes MLP-M.** MLP-M is explicitly a baseline in the protocol and beats CAP* in worst-node accuracy; the ten Family B comparisons exclude it.

  Replace with:
  > CAP* (= CAP0.1 in every cell) beats each of the ten Family B comparators on worst-node@20% in 15/15 cells. MLP-M is excluded from Family B and remains more robust than CAP*.

- **SHOULD-FIX — lines 87–92: the CAP-versus-ENS conclusion is stronger than the exploratory evidence.** The label is present, but “matched blend” suggests matching that was never performed. These are selected nearby mean-clean comparisons; two of the three robustness CIs include zero. ENS0.1’s reverse comparisons are already acknowledged, which is fair.

  Replace the heading with:
  > **Selected comparisons with nearby mean-clean ENS settings (exploratory, uncorrected):**

  Replace line 91 with:
  > These selected comparisons suggest a possible advantage over some tested blends, but do not establish that CAP beats blending at matched clean accuracy. Two of the three robustness CIs include zero; ENS is uncalibrated, and CAP also trains its correction through the combined output.

- **SHOULD-FIX — lines 93–94: exploratory nonsignificance becomes a substantive conclusion.** RES versus ENS0.25 has robustness CI **[−6.57, +4.81]**. “No better” implies an established absence of benefit; “the cap is what helps” then uses that exploratory comparison as evidence, contrary to the protocol’s restriction.

  Replace with:
  > **RES vs ENS0.25 (exploratory, uncorrected):** mean clean difference −0.23 points; mean worst-node difference −0.88 [−6.57, +4.81]. This comparison establishes neither an advantage nor equivalence. The earlier confirmatory H82 supports CAP* over RES, with lower clean accuracy; it does not establish CAP over blending.

- **SHOULD-FIX — lines 105–106: the attack-strength caveat has the direction reversed.** Optimistically high RUNG-lite robustness favors RUNG-lite in the observed comparison and makes CAP’s envelope test harder. Correcting that optimism could favor CAP.

  Replace with:
  > RUNG-lite is a simplified reimplementation evaluated without an additional RUNG-specific adaptive attack. Its measured 49.9 worst-node accuracy may therefore overestimate its robustness, making the current comparison harder for CAP. Stronger attacks could change the high-clean comparison.

- **SHOULD-FIX — lines 19–21: “strongest baseline” needs a criterion.** MLP-M and ENS0.1 have higher mean worst-node accuracy. The RUNG-lite comparison is otherwise numerically fair, and the wide paired CI is appropriately disclosed later.

  Replace with:
  > At CAP0.75 and CAP0.9, the mean eligible-envelope gaps are significantly negative. RUNG-lite is a strong high-clean baseline (81.7 clean, 49.9 worst-node): its mean robustness is close to CAP0.75’s, with 5.42 points more clean accuracy. Their exploratory paired robustness difference, CAP0.75 minus RUNG-lite, is +0.35 [−7.51, +8.21], which does not establish equivalence.

- **SHOULD-FIX — lines 109–110: clarify the inferential limitation.** The existing dataset caveat is useful, but the t intervals treat 15 cell differences as independent; shared graph and overlapping splits make nominal coverage uncertain. Holm does not repair this.

  Add:
  > The paired t tests use 15 cell differences as independent observations. Shared graph structure and overlapping split information may induce dependence, so nominal CI coverage and p-values should be interpreted cautiously. Holm correction addresses multiplicity, not this dependence.

- **NIT — lines 54–60: identify the CIs as unadjusted.** The verdicts use corrected p-values correctly, but the displayed intervals are ordinary marginal intervals.

  Replace the column heading with:
  > Mean D [unadjusted 95% CI]

Did the capped correction find a new frontier relative to the baselines? **It added promising low-clean trade-off points relative to the tested baseline grid.** CAP0.1 and CAP0.25 beat the prespecified per-cell eligible baseline envelope on average, with multiplicity-corrected significance. That supports two tested points, not a continuous or generally superior frontier: baseline mean-clean coverage is sparse there, finer or calibrated blends were not tested, and MLP-M remains more robust. CAP0.5 is inconclusive, while the higher caps fall below the eligible envelope.