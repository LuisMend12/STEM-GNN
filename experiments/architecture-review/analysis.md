# Analysis E1: raw edge-text injection (finding F1a) (reviewed by Codex; corrections applied)

**Setup.** Real `Encoder`, non-MoE SAGE, hidden width 384 = input dim, BatchNorm, OFA Cora with the constant Cora edge text c (‖c‖ = 1.0 = ‖x_j‖). 5 seeds × 3 splits (n = 15 paired). Threads = 4, with `use_deterministic_algorithms(True, warn_only=True)`.

**Raw data:** `results/edge_injection_split{0,1,2}.json` (120 rows).

## Results (test accuracy %, mean ± SD, n = 15)

| Variant | Depth 2 | Depth 4 |
|---|---|---|
| S0: no edge features | 78.5±1.6 | 78.3±2.4 |
| **S1: repo behavior (raw c, every layer)** | 77.7±1.7 | 78.6±2.3 |
| S2: raw c at layer 0 only | 77.9±1.5 | 78.8±2.1 |
| S3: learned per-layer projection (EdgeProj) | 77.5±3.0 | 77.2±3.3 |

Paired differences (mean, SE):

| Comparison | Depth 2 | Depth 4 |
|---|---|---|
| S1 − S0 | −0.79 (0.51) | +0.23 (0.55) |
| S2 − S1 | +0.16 (0.32) | +0.20 (0.18) |
| S3 − S1 | −0.27 (0.76) | −1.39 (0.88) |
| S3 − S0 | −1.06 (0.59) | −1.15 (1.09) |

H26 quantity, (S2 − S1) at depth 4 minus (S2 − S1) at depth 2: +0.04 (SE 0.39).

## Hypothesis outcomes

| ID | Rule | Result | Verdict |
|---|---|---|---|
| **H24** | S1 − S0 at depth 2: ≤ −1 harmful, ≥ +1 helpful, otherwise no effect | −0.79 | **No effect** under the rule. The point estimate leans negative and is not resolved. |
| **H25** | S3 − S1 ≥ +1 at depth 2 or 4; refuted if ≤ 0 at both | −0.27, −1.39 | **Refuted** |
| **H26** | difference ≥ +1 | +0.04 | **Not supported** |

## Interpretation (revised per Codex review)

**Headline:** this Cora proxy did not cross the pre-registered harm threshold, and the learned edge projection did not improve the point estimates. Neither result establishes equivalence or generalizes to informative edge attributes.

1. **Verdicts are threshold classifications, not equivalence tests.** A paired-t interval for depth-2 S1−S0 is about [−1.88, +0.30], which still allows harm larger than 1 point. The 15 runs share one graph and three splits, so the SE describes runs, not 15 independent datasets.
2. **The "absorption" mechanism is untested.** The constant shift sits *inside* the message ReLU, so BatchNorm or later affine maps cannot be assumed to cancel it exactly.
3. **S3 is confounded.** It adds about 295k parameters at depth 2 and 591k at depth 4, which with a constant edge feature are a redundant per-layer message bias. A directly learned bias would be a cleaner control. Its extra initialization also consumes RNG, so its dropout draws are not matched to the other variants.
4. **Feature pipeline differs from the repo.** The experiment uses raw `d.x` and MiniLM edge embeddings, whereas the repo regenerates prefixed node-text embeddings with its configured encoder. "What the repo feeds" refers to the edge-text content and the injection rule, not identical numerical inputs. Feature norms and package versions are not recorded in the result files.
5. **What this cannot answer:** whether MoE discarding *informative* edge features (F1b) matters on molecules. That needs the molecule dataset plus `rdkit`.
6. **Protocol wording mismatch:** the protocol states H26 as strict `> +1` and the table above writes `≥ +1`. The verdict is unaffected.

## Threats to validity

- One dataset.
- Constant edge feature.
- MiniLM-384 in place of the repo's 768-d encoder.
- Depth ≤ 4.
- Determinism was requested with `warn_only`, so some operations may still be nondeterministic.
