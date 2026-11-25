I reran `analyze.py` against all 65 JSON files. The numerical tables, paired differences, confidence intervals, win counts, Holm-adjusted p-values, and guardrail results reproduce the supplied output. The H100–H104 verdicts and the two protocol claim verdicts follow the stated rules. The main problems are interpretations that go beyond those results.

- **BLOCKER — [analysis.md:19](<G:/broswer download/STEM-GNN/experiments/vgae-weighting/analysis.md:19>) and [analysis.md:127](<G:/broswer download/STEM-GNN/experiments/vgae-weighting/analysis.md:127>): The headline and conclusion overstate the adaptive result.** H103 meets the protocol’s operational rule, but the tested-attack result does not establish robust worst-case performance or zero clean cost. The SAGE result is descriptive, and a larger own-attack score raises an unresolved attack-strength concern.

  Replace lines 19–24 with:

  > **Adaptive edge attacks (evasion): the prespecified STEM comparison meets its operational claim.**
  > - S-VG − S0 worst-node accuracy over the tested attack union at 20% is +3.09 [+1.47, +4.70] points, with 13/15 cell wins and Holm p = .005 (H103 supported).
  > - Clean accuracy changes by −0.34 [−1.65, +0.97] points, meeting the prespecified guardrail; this does not establish zero clean cost.
  > - The analogous SAGE-VG − SAGE difference is descriptive: +4.30 [+3.77, +4.83] points on the tested union, with a rounded clean difference of +0.00 [−0.23, +0.24].
  > - Attack-strength and gradient-masking concerns limit the robustness interpretation.

  Replace lines 127–131 with:

  > These results identify a candidate direction for further study: structure-only edge weighting fitted on a clean graph. On this Cora setup, the prespecified STEM comparison meets its operational adaptive-attack claim, while the GraphSAGE result is descriptive. The random-injection claim is not met, and the tested-suite gains do not establish a new robustness frontier.

- **SHOULD-FIX — [analysis.md:25](<G:/broswer download/STEM-GNN/experiments/vgae-weighting/analysis.md:25>): The poisoning-versus-evasion explanation is post hoc and confounded.** The text acknowledges that it was not tested directly, but calls it a “mechanism” and attributes the AUC trend to increasing noise. The experiments also change the attack type, graph, training regime, seeds and evaluation; the AUC comparison across levels is descriptive and unpaired.

  Replace lines 25–32 with:

  > **A post hoc hypothesis, not a tested mechanism:**
  > - In Part R the VGAE is fitted on each corrupted graph; mean original-versus-injected edge AUC ranges from 0.750 at 500 edges to 0.667 at 5000, below structural Jaccard at each level.
  > - In Part A the VGAE is fitted on the clean graph before adaptive evasion attacks.
  > - These observations motivate testing whether clean versus corrupted VGAE training matters, but the two parts also differ in attack type, model training and evaluation. The AUC levels use different graphs and seeds, so neither the mechanism nor a causal noise effect is established.

- **SHOULD-FIX — [analysis.md:109](<G:/broswer download/STEM-GNN/experiments/vgae-weighting/analysis.md:109>): The masking caveat needs a sharper interpretation.** S-VG’s 53.8 versus S0’s 46.6 is accuracy on *different, model-targeted attack sets*, not proof that one attack is intrinsically weaker. The shared union and SAGE transfer provide useful cross-model tests, but neither establishes that the weighted model has been adequately attacked.

  Replace lines 109–115 with:

  > - **The adaptive result is limited to the tested attack union.** Own-attack worst-node accuracy is 53.8 for S-VG versus 46.6 for S0, but those figures use different model-targeted attack sets. The gap raises concern that PRBCD may optimize less effectively against the weighted model; it does not by itself prove gradient masking.
  > - The shared union includes attacks targeted at unweighted S0, and SAGE-transfer accuracy is 69.1 for S-VG versus 65.5 for S0. These are useful transfer checks, but neither rules out a stronger attack tailored to S-VG.
  > - Further attack validation, including alternative gradients such as BPDA and stronger optimization or more restarts, is needed before interpreting the gain as reliable adaptive robustness. No certificate was produced.

- **SHOULD-FIX — [analysis.md:98](<G:/broswer download/STEM-GNN/experiments/vgae-weighting/analysis.md:98>) and [analysis.md:102](<G:/broswer download/STEM-GNN/experiments/vgae-weighting/analysis.md:102>): The missing-own-attack explanation is correct, but the comparison is too strong.** `run.py` includes clean in the own-attack stack, so for unattacked arms the stored value equals clean. Omitting an S-SJ-targeted attack can only leave its measured union score unchanged or higher than adding that attack would; it does not establish an overall fair or conservative comparison between methods.

  Replace lines 98–99 and 101–103 with:

  > For unattacked arms, no own-targeted attack exists: the script’s own-attack calculation contains clean alone and therefore returns clean accuracy. The table shows “—” rather than treating that value as an attack result. MLP-M is also unattacked; its reported value remains 70.0 for every graph because it ignores edges.
  >
  > **Descriptive (uncorrected):**
  > - S-VG − S-SJ worst-node@20% on the shared tested union is +2.21 [+1.12, +3.30], with 12/15 cell wins. S-SJ was not directly attacked. Adding an S-SJ-targeted graph could only maintain or lower its score on an expanded union, but this comparison does not establish its robustness against a tailored attack or resolve attack-strength differences between methods.

- **SHOULD-FIX — [analysis.md:14](<G:/broswer download/STEM-GNN/experiments/vgae-weighting/analysis.md:14>) and [analysis.md:16](<G:/broswer download/STEM-GNN/experiments/vgae-weighting/analysis.md:16>): “No gain” and “doesn’t beat” turn inconclusive results into negative conclusions.** H101 and H102 are explicitly inconclusive; H102’s interval still includes positive effects.

  Replace lines 14–18 with:

  > **Random edge injection (poisoning): the protocol claim is not met.**
  > - S-VG − S0 is +0.03 at 2000 edges (H100: evidence against under the operational rule) and +0.51 at 5000 (H101: inconclusive).
  > - S-VG − untrained VGAE is −0.00 at 5000 edges (H102: inconclusive).
  > - SAGE-VG − SAGE is +0.29 at 5000 edges (H104: evidence against under the operational rule).

- **SHOULD-FIX — [analysis.md:36](<G:/broswer download/STEM-GNN/experiments/vgae-weighting/analysis.md:36>) and [analysis.md:69](<G:/broswer download/STEM-GNN/experiments/vgae-weighting/analysis.md:69>): Two unsupported generalizations.** “Cheap” has no reported timing or resource comparison; “Macro-F1 follows the same pattern” is not quantified and can conceal differences between metrics.

  Replace line 36 with:

  > S-VG’s measured gain is relative to S0 on this attack suite; RUNG-lite and MLP-M score higher in this setup.

  Replace lines 69–70 with:

  > Macro-F1 results are reported in `analysis_output.txt` and are descriptive. Each noise level uses its own corruption and pretraining seeds, so comparisons across levels are descriptive; paired arm comparisons are within a level.

- **NIT — [analysis.md:52](<G:/broswer download/STEM-GNN/experiments/vgae-weighting/analysis.md:52>): Clarify what the intervals and threshold mean.** The write-up identifies shared splits, but should state that the support rule does not test whether the true improvement reaches two points.

  Replace lines 52–53 with:

  > These operational decisions are conditional on Cora. Cells share one graph; Part A’s 15 cells share three splits, and Part R uses splits 4/3/3. The intervals describe run-level variation, not independent-dataset replication. The +2-point support threshold applies to the estimate; the significance test is against zero, not against +2.

The own-attack table’s dashes are therefore appropriate, and the SAGE-VG adaptive comparison is explicitly labeled descriptive where first reported. The protocol’s S-VG adaptive claim is met, but that is a narrower statement than proving the method robust to adaptive attacks generally.

This is promising evidence for a specific research direction, not yet evidence of a new frontier. On one Cora setup, structure-only VGAE weighting improved STEM-GNN’s score by 3.09 points over the tested adaptive attack union and met the prespecified clean-accuracy guardrail. The analogous GraphSAGE result is descriptive, and the random-poisoning claim was not met. The adaptive gain needs stronger attack validation because the weighted model’s own-targeted attacks may be less effective, even though transfer attacks were included. Testing the mechanism directly, comparing against a directly attacked Jaccard control, and replicating on other datasets would show whether the result extends beyond this setting.