# Analysis: X1 (MoE per-expert init) and X2 (codebook update rule) (reviewed by Codex — `../reviews/codex_stem_fixes_analysis_review_2026-10-27.md`; corrections applied)

**Raw data:** `results/x{1,2}_split{0,1,2}_main-20261026.json` (90 X1 rows, 60 X2 rows).

**Setup.**

- STEM-GNN's real `Encoder` and `VectorQuantize`, on OFA Cora.
- 5 seeds × 3 splits (n = 15 paired, conditional on this graph and these splits).
- **Threads 2**, which differs from the protocol's 4. Deterministic algorithms requested (`warn_only`).
- A pre-launch fresh-process repeatability check matched exactly, but its outputs are not archived with these results.
- X2 observations average 3 probe seeds before inference. Intervals summarize variation across the 15 seed/split pairs on this graph, not independent datasets.
- Uncertainty: paired t, 95% CI with t(14) = 2.145. Accuracies are in percentage points; SD is the sample SD.

## X1: per-expert initialization of MoE layers (width 128)

| Variant | Depth 2 | Depth 16 | depth_drop = acc(2) − acc(16), mean (SE) |
|---|---|---|---|
| SAGE | 76.28±2.22 | 71.11±3.46 | 5.17 (0.96) |
| **M-stack (repo)** | **80.66±1.59** | **74.85±3.08** | 5.80 (0.80) |
| M-expert (fixed init) | 79.43±2.05 | **47.23±14.95** | **32.20 (4.14)** |

| Paired contrast | Mean | SE | 95% CI |
|---|---|---|---|
| **H27 depth_benefit** = drop(M-stack) − drop(M-expert) | **−26.40** | 4.43 | [−35.90, −16.90] |
| M-expert − M-stack, depth 2 | −1.23 | 0.47 | [−2.23, −0.23] |
| M-expert − M-stack, depth 16 | −27.62 | 4.11 | [−36.44, −18.81] |

Per-split depth_benefit: −26.47, −34.21, −18.51.

**Diagnostics** (means over 15 runs):

| Variant | Depth | Mixed transform/residual ratio, at init | After training | Entropy of mean routing | Mean per-node entropy |
|---|---|---|---|---|---|
| M-stack | 2 | 0.067 | 1.53 | 0.78 | 0.76 |
| M-stack | 16 | 0.068 | 3.26 | 0.88 | 0.78 |
| M-expert | 2 | 0.621 | 1.87 | 0.87 | 0.84 |
| M-expert | 16 | 0.649 | 4.21 | 0.86 | 0.73 |

**Verdicts (Amendment 1 rules):**

| ID | Verdict | Evidence |
|---|---|---|
| H27 (primary) | **Refuted, and strongly reversed** | −26.4, CI excludes 0, every split negative |
| H28 | **Not supported; strong opposite-direction evidence** | −27.6, CI [−36.44, −18.81]. No formal refutation rule was registered. |
| H29 (non-inferiority at depth 2) | **Fails the registered criterion; non-inferiority not established** | −1.23, CI [−2.23, −0.23]. The CI crosses −1, so a true loss larger than 1 point is not established either. |
| H30 (manipulation check) | **Passed** | the mixed ratio rises 9.23× at depth 2 and 9.61× at depth 16 (≥ 5×); the theoretical weight-RMS multiplier is 9.29×. This is a manipulation check, not a causal explanation. |

**Interpretation (revised per Codex).**

- **Under the registered width-128 Cora recipe, stacked initialization substantially outperforms per-expert Xavier at depth 16.** That is an initialization-dependent performance difference under this optimizer (Adam, lr 0.01) and stopping rule. It does **not** establish *why* the larger initialization fails: no loss, gradient or activation trajectories were archived, and no LR control was run. M-expert depth-16 runs averaged 146 epochs vs 113 for M-stack, so they did not simply stop immediately.
- **Small stacked weights suppress the initial transform relative to the residual** in equal-width layers. That resembles the motivation for small residual-branch initialization, but equivalence to ReZero, SkipInit or Fixup, and mediation of the accuracy effect, are untested.
- **The manipulation is broader than residual scaling.** It changes expert draws and scales in every MoE layer, including the non-residual first layer. No `per_expert_init` or `fork_rng` bug was found.
- **Architecture-review F2 is revised, not reversed.** The fan calculation stays correct; the proposed blanket fix is withdrawn. Initialization scale is a design choice that needs joint optimizer validation, not an established defect.
- **Depth gap at width 128:** the estimated excess MoE depth drop over SAGE is only 0.63 points. That does not establish equivalence or identify width as the cause of differences from `../moe-depth`.
- **Next discriminator (proposed, not run):** a depth-16 factorial of both initializations × lr {0.01, 0.001} (optionally with warmup), logging loss, gradient norms, pre-BN activation RMS and BN running variances. Then a same-draw scale ablation on the residual layers only.

## X2: codebook update rule (STEM-component VQ pilot)

| Arm | Code-probe test | z-probe test | Code perplexity (of 128) | Dead-code fraction | Final feat / topo loss |
|---|---|---|---|---|---|
| C-repo (`ema=F`, `ortho=1`) | 75.07±1.37 | 77.21 | 13.6 | 0.753 | 0.0018 / 0.865 |
| C-random (`ema=F`, `ortho=0`) | 75.31±1.41 | 77.10 | 15.1 | 0.688 | 0.0018 / 0.864 |
| **C-ema** (`ema=T`, `ortho=0`) | **77.46±1.24** | 76.58 | **94.8** | **0.124** | 0.0018 / 0.883 |
| C-none (no VQ) | 79.82±1.18 | 79.82 | — | — | 0.0042 / 0.894 |

| Paired contrast (code-probe) | Mean | SE | 95% CI |
|---|---|---|---|
| **H31: C-ema − C-random** | **+2.15** | 0.54 | **[+0.98, +3.31]** |
| C-ema − C-repo | +2.39 | 0.54 | [+1.24, +3.54] |
| C-random − C-repo | +0.24 | 0.37 | [−0.55, +1.03] |
| C-none − C-ema (descriptive reference gap) | +2.36 | 0.41 | [+1.48, +3.24] |
| z-probe, C-ema − C-random | −0.52 | 0.53 | — |

Per-split H31 means: +1.09, +1.83, +3.52.

**Verdicts:**

| ID | Verdict | Evidence |
|---|---|---|
| H31 (primary) | **Supported** | +2.15 ≥ +2, and the CI excludes 0. It is marginal: two of three split means are below +2. |
| H32 | **Supported** | perplexity 94.8 vs 15.1, dead codes 12% vs 69% |
| H33 | Continuous-reference gap of +2.4 to C-ema (+4.5 to C-random) | descriptive only |

**Interpretation (revised per Codex).**

- **EMA improves code-probe accuracy by +2.15 points** (CI [+0.98, +3.31]). This compares equal-dimensional (512-d) probes with matched initialization and corruption schedules. Split means are +1.09, +1.83 and +3.52, so consistency is limited, and the CI does not establish a true effect ≥ 2.
- **The z-probe difference is unresolved:** −0.52, CI [−1.65, +0.61]. EMA changes the codebook and the encoder's training targets together. Greater code utilization accompanies the gain, but nothing shows that utilization mediates it, or isolates codebook fitting from encoder–codebook co-adaptation.
- **Utilization:** mean per-head perplexity rises from 15.1 to 94.8 (6.27×; every head about 95 vs about 15). The fraction of codes unused on clean evaluation falls from 68.8% to 12.4%, so occupied codes rise 2.81×. These are occupancy diagnostics, not persistent code death. Dead-code replacement is disabled (expiry threshold 0).
- **The orthogonality comparison is unresolved:** C-random − C-repo is +0.24, CI [−0.55, +1.03]. No equivalence is claimed.
- **Continuous-reference gaps (C-none − arm):** +4.75 vs C-repo, +4.51 vs C-random, +2.36 vs C-ema. C-none is a separately trained reference, not an estimate of quantization's information loss.
- **Decay 0.8 was fixed in advance;** its sensitivity is untested.
- **Scope:** a feature + topology reconstruction pilot, not STEM's full objective.

## What this means for robustness

**Neither X1 nor X2 is a robustness method,** and neither was tested under attack. They are architecture findings:

- X1 shows that per-expert Xavier, without retuning the optimizer, is harmful at depth.
- X2 shows that an EMA-fitted codebook improves code-probe accuracy in this pilot.

## Threats to validity

- One dataset.
- Width 128 (X1); repo default is 768.
- One codebook size (128).
- Pilot pretraining objective (X2).
- Linear probes rather than full finetuning.
- `warn_only` determinism.
