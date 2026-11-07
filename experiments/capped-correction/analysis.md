# Analysis: capped graph correction against edge attacks

Protocol: `protocol.md`. Code: `STEM-GNN/scripts/capped/run.py`, `analyze.py`. Raw output:
`analysis_output.txt` (run on the cluster, Slurm 32888163). Run: Slurm array 32738894, all 15 cells (Cora,
splits 0–2 × seeds 0–4) completed with no errors; one configuration hash (`43af6ed6a6`). The selfcheck
(27 checks) passed beforehand.

## Bottom line

- **Claim not met.** The prespecified claim needed H81 (clean gain over MLP-M ≥ +2) and H82 and the S0
  gate. H82 and the gate are met; H81 isn't (+1.03).
- **H80 is against:** under the tested attacks, the selected CAP\* is 1.67 points *worse* than the
  graph-free MLP-M at 20% worst-node (0/15 cells better). So CAP\* doesn't beat simply ignoring the graph.
- **The selection rule collapsed to the smallest cap** in all 15 cells. CAP\* is MLP-M plus a small
  correction, keeping 12% of S0's clean gain over MLP-M (ratio of means).
- **What did work, descriptively:** the cap grid traces a clean-versus-robust curve between S0 and MLP-M,
  and every capped point is far more robust than S0 or the uncapped residual. Whether that curve beats
  standard baselines or a plain S0/MLP-M ensemble hasn't been tested; that's the follow-up
  (`baselines_protocol.md`).

## Prespecified results (n = 15, two-sided paired t, points, Holm over H80–H82)

| ID | Comparison | Endpoint | Mean [95% CI] | Wins | Holm p | Verdict |
|---|---|---|---|---|---|---|
| H80 | CAP\* − MLP-M | Worst-node@20% | −1.67 [−2.13, −1.21] | 0/15 | <.0001 | Against (upper CI < 0) |
| H81 | CAP\* − MLP-M | Clean | +1.03 [+0.70, +1.36] | 14/15 | <.0001 | Inconclusive: significant, but the mean is below the +2 support threshold and the upper CI (+1.36) is not below +1 |
| H82 | CAP\* − RES | Worst-node@20% | +25.42 [+17.39, +33.45] | 13/15 | <.0001 | Supported |
| Gate | CAP\* − S0 | Worst-node@20% | +33.48 [+30.93, +36.03] | 15/15 | — | Met |

Claim rule (H81 and H82 and gate): **not met**. Because H80 is against, the honest reading is a trade-off:
at the selected cap, CAP\* has lower worst-node accuracy than MLP-M and slightly higher clean accuracy.

## Descriptive

Arm means over 15 cells (points):

| Arm | Clean | Worst-node@20% | Certified | Worst-node@10% | Own-attack worst-node@20% | SAGE transfer@20% |
|---|---|---|---|---|---|---|
| S0 | 78.3 | 34.8 | — | 44.3 | 46.8 | 65.8 |
| RES (uncapped) | 76.8 | 42.9 | — | 50.1 | 53.4 | 68.0 |
| CAP0.9 | 76.5 | 45.9 | 9.7 | 52.2 | 52.2 | 69.3 |
| CAP0.75 | 76.2 | 51.5 | 23.0 | 56.3 | 56.8 | 70.8 |
| CAP0.5 | 75.0 | 57.6 | 43.1 | 60.7 | 61.0 | 71.0 |
| CAP0.25 | 72.7 | 64.9 | 58.8 | 66.4 | 66.7 | 71.1 |
| CAP0.1 = CAP\* | 71.0 | 68.3 | 65.9 | 68.8 | 69.0 | 70.5 |
| MLP-M | 70.0 | 70.0 | — | 70.0 | 70.0 | 70.0 |

- CAP\* − S0 clean: −7.33 [−8.78, −5.87].
- Attack slack for CAP\* (worst-node − certified): +2.35 [+1.83, +2.86]. For this nodewise metric, the true
  worst case lies between the certified floor and the measured value, so at this cap at most about 2 points
  of measured robustness could be an artefact of weak attacks. It doesn't show PRBCD is near-optimal.
- For larger caps, the gap is large (CAP0.75: 51.5 versus 23.0). Those worst-node numbers are accuracy
  against the tested attacks only; stronger attacks *may* lower them, and the certificate isn't
  necessarily attainable either.
- RES − S0 worst-node@20%: +8.06 [+1.67, +14.45], with high variance (RES sd 15.6). Residual correction
  improves robustness descriptively; the selected capped correction improves it substantially further
  (H82), with lower clean accuracy.
- Per-split CAP\* − MLP-M worst-node: −1.77, −1.20, −2.04; per-split CAP\* − S0 clean: −7.02, −5.44, −9.53.
  The direction is the same in every split.

## Why CAP\* landed on MLP-M

The selection score is mean(clean val, certified val), computed per cell on validation. Mean validation
values across the 15 cells:

| Cap | Clean val | Certified val | Score |
|---|---|---|---|
| CAP0.1 | 70.8 | 64.9 | 67.9 |
| CAP0.25 | 72.7 | 57.6 | 65.2 |
| CAP0.5 | 75.2 | 42.9 | 59.0 |
| CAP0.75 | 76.4 | 23.0 | 49.7 |
| CAP0.9 | 78.0 | 9.5 | 43.7 |

Certified validation accuracy falls about 55 points across the grid while clean validation accuracy rises
about 7, so the smallest cap won in every cell (winning margin over the runner-up at least 1.3 points, mean 2.7).
This was a choice made in the protocol, not a result about the method. Other caps are reported above only
descriptively; picking one now would be post hoc, and none is claimed.

## What this does and doesn't show

- It shows that, under this suite, capping the graph correction adds a large robustness gain beyond the
  uncapped residual (H82). Every capped point's mean worst-node@20% is 11–34 points above S0's.
- It doesn't show a method that beats the graph-free model under attack: the certificate can't exceed
  MLP-M's accuracy, and the measured worst-node didn't either, at any cap.
- It doesn't yet show that CAP's curve beats standard baselines or an ensemble of S0 and MLP-M at matched
  clean accuracy. That comparison is specified in `baselines_protocol.md`, written before it runs.
- Scope: Cora only, edges only, a PRBCD attacker that uses test labels; worst-node for larger caps is an
  upper bound on true robustness. The 15 cells share one dataset (3 splits × 5 seeds, overlapping test
  information), so the CIs describe training and split variability on Cora, not replication across
  datasets. The direction of every confirmatory effect is the same in all three splits. No novelty claim (the protocol requires a prior-art check first, and the
  result doesn't call for one yet).
