# Analysis: why STEM-GNN's MoE encoder degrades with depth

Reviewed by Codex (`../reviews/codex_moe_depth_collapse_review_2026-10-22.md`). Its corrections are applied and each was verified against the raw JSON.

**Setup.** The real `Encoder`, MoE with K=3 and `moe_layers='all'`, BatchNorm, hidden width 64, OFA Cora. 5 seeds × 3 splits (n = 15 paired by split and seed), threads fixed at 4.

**Raw data:** `results/split{0,1,2}.json`, 120 rows. Standard deviations are sample SDs. Paired SEs are the sample SD of the 15 matched differences divided by √15. All tabulated values were recomputed by Codex from the raw JSON and match.

## Results (test accuracy %, mean ± SD, n = 15)

| Variant | Depth 2 | Depth 16 | Drop (16 − 2), mean (SE) |
|---|---|---|---|
| SAGE (reference) | 76.2±3.9 | 74.5±3.1 | −1.67 (1.10) |
| **V0** MoE, Gumbel train / softmax eval (repo default) | 80.0±1.9 | 74.3±3.7 | **−5.74 (0.89)** |
| V1 = V0 model, Gumbel-sampled eval (16 draws, mean logits) | 80.1±1.9 | 74.6±3.7 | −5.49 (0.94) |
| V2 MoE, softmax train (no Gumbel noise) | 79.1±1.6 | 75.5±2.3 | −3.54 (0.68) |
| V3 MoE, uniform frozen routing | 79.1±1.7 | 75.3±2.0 | −3.85 (0.62) |

Routing TV for V0 is similar at the two depths: 0.283 and 0.287.

Paired differences (mean, SE, n = 15):

| Comparison | Depth 2 | Depth 16 |
|---|---|---|
| V1 − V0 | +0.10 (0.06) | +0.36 (0.27) |
| V2 − V0 | −0.91 (0.37) | +1.29 (1.23) |
| V3 − V0 | −0.90 (0.38) | +1.00 (1.17) |
| V0 − SAGE | +3.80 (0.84) | −0.27 (1.35) |

| Drop comparison | Mean (SE) |
|---|---|
| drop(V2) − drop(V0) | +2.20 (1.17) |
| drop(V3) − drop(V0) | +1.89 (1.25) |
| drop(V3) − drop(SAGE) | −2.18 (1.09) |

## Hypothesis outcomes (pre-registered point-estimate rules)

| ID | Rule | Result | Verdict |
|---|---|---|---|
| **H20** | support ≥ +1.5; refute ≤ 0 | +0.36 | **Inconclusive** (strictly between the thresholds) |
| **H21** | support ≥ +3; refute < +1 | +2.20 | **Inconclusive** |
| **H22** | support ≥ +3; refute ≤ +1 | +1.89 | **Inconclusive.** Even a refutation would not uniquely identify architecture as the cause. |
| **H23** | each \|Δ\| at depth 2 ≤ 1 | +0.10, −0.91, −0.90 | **Supported under the point-estimate rule.** This is not a demonstrated equivalence or proof of depth specificity. |

## Interpretation

1. **Eval-time Gumbel sampling recovers little.** Sixteen-draw Gumbel mean-logit inference recovers an estimated +0.358 points at depth 16 (SE 0.268). That doesn't reach the pre-registered 1.5-point effect, but H20 stays inconclusive, and this intervention does not isolate every consequence of stochastic training.
2. **Removing Gumbel noise during training may shrink the drop.** The estimate is +2.2 (SE 1.2), about 1.9 SE from zero, so it is unresolved.
3. **Uniform-routing MoE still loses accuracy with depth.** Its paired drop exceeds SAGE's by 2.18 points (SE 1.09). That comparison does not isolate a causal architectural mechanism: V3 stays above SAGE at both depths, and its larger drop includes the loss of its shallow advantage (+2.9 at depth 2, +0.7 at depth 16).
4. **Joint vs separate weights is not an expressive difference.** A joint linear map on concatenated neighbor and root features is algebraically equivalent to separate linear maps. Uniform expert averaging changes parameterization, initialization and optimization, and these must be controlled before attributing the depth difference to an architectural operation.
5. **Other differences between the two paths:**
   - bias (SAGE's neighbor map has one, MoE experts do not);
   - initialization;
   - expert parameterization;
   - the residual is added *before* BatchNorm;
   - residuals are absent in the first 384→64 layer but present in every later layer, including the last.

   Edge attributes are ignored by MoE, and aggregation direction differs, but neither matters in this undirected, attribute-free harness. With ReLU and non-negative dropout outputs between layers, SAGE's message ReLU mainly changes inputs at the first layer.
6. **Uncertainty.** Depth-16 accuracy differences are small relative to their uncertainty, and similar means do not establish equivalence. Precision varies by comparison, so there is no single two-point resolution limit.

## Reproducibility finding (important)

**Deep MoE runs are not reproducible at a fixed seed and thread count.** The depth-16 MoE baseline (V0) differs from the identically configured `message-collapse` MoE baseline in **15 of 15** (split, seed) pairs, by up to **5.85 points**. SAGE at both depths and MoE at depth 2 match exactly (15/15).

- **Implications.** Shared seed labels do not align stochastic trajectories, and the two baseline sets must not be pooled or treated as exact replications. Neither script enables deterministic algorithms.
- **Likely sources.** Nondeterministic parallel scatter reductions compounding over 16 MoE layers, or different random-stream consumption once routing draws Gumbel noise.
- **Status:** the cause is unresolved. This is also why V2 and V3 (no Gumbel draws) are not trajectory-matched to V0.

## Mechanics verified (Codex)

- V2/V3 replace only the encoder module's `F` reference and restore it in a `finally` block.
- V3's frozen zero router stays uniform despite Adam weight decay.
- V1 samples softmax((logits + Gumbel)/τ) with dropout and BatchNorm in eval mode. The reported TV is valid for τ = 1 and is computed at sampled-path router inputs; it is not an end-to-end discrepancy between deterministic and sampled trajectories.
- V1's reseeding does not affect later variants.
- These mechanics assume variants run sequentially within each process.

## Protocol integrity

The amendment repairs a Gumbel-noise implementation error without changing hypotheses or variants. The earlier V0 accuracy and the broken V1 diagnostic had nevertheless been observed, and the deleted partial results cannot be reconstructed.

## Next decisive experiment (proposed by Codex, not yet run)

**Matched effective-operator test: SAGE vs uniform MoE at depths 2 and 16.**

1. Express uniform MoE as its exact averaged neighbor and root matrices, absorbing the residual into the root matrix.
2. Match SAGE's first-layer message ReLU, bias, initialization, BatchNorm placement, dropout masks and optimizer/weight-decay treatment.
3. Optimize the effective matrices directly.
4. Verify that forward outputs and gradients match before training.
5. Run with deterministic algorithms enabled, on fresh seeds.

**Pre-registered prediction:** the extra uniform-MoE depth penalty disappears within an equivalence margin of ±1 point. A remaining penalty would mean another implementation difference must be found.

## Threats to validity

- One dataset.
- Width 64.
- No pretraining.
- BatchNorm on.
- Only τ = 1 was tested.
- Nondeterministic deep-MoE training.
- Novelty was checked with two web searches only.
