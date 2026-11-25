**NO-GO for the full experiment.** The revision fixes several design issues, including the RUNG qualification, but exact reproduction is not yet guaranteed and the resume files lack sufficient binding to their checkpoint. Resolve the blockers below before the declared checks or grid.

The RUNG wording at [protocol.md:15](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:15>) is accurate: the **added VGAE input weights** use structure, while RUNG’s own prediction-dependent propagation coefficients use text features. Given the explicit request to put VGAE weights inside RUNG-lite, the qualified design is appropriate.

## New findings

### BLOCKER — Original `worst_graph` reduction is not reproduced exactly

[followup.py:470](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:470>) takes the minimum of separately computed one-dimensional means. The original [run.py:364](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/run.py:364>) computes a two-dimensional tensor reduction. Those mathematically equivalent reductions can use different floating-point accumulation orders, so they need not produce the exact JSON float. The same issue affects the expanded `worst_graph` at line 460.

Replace both calculations with the original operation:

```python
suite = torch.stack([clean] + [corr[k] for k in keys])
row[f"worst_graph@{r}"] = suite[:, te].float().mean(1).min().item()
```

And, for the original union:

```python
old_suite = torch.stack([clean] + [corr[k] for k in ok])
row[f"old_worst_graph@{r}"] = old_suite[:, te].float().mean(1).min().item()
```

The other old-union definitions use the original inclusive `k[2] <= r`, clean inclusion, own-arm filter, and exact-budget SAGE transfer semantics. The strict comparison in [analyze_followup.py:90](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/analyze_followup.py:90>) is appropriate for those declared metrics, but it has not yet established that they reproduce.

### BLOCKER — Resumed graph files are accepted without provenance or validation

[followup.py:428](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:428>) trusts any file at the expected path, loading neither a graph key nor a configuration/checkpoint identity from it and not revalidating its graph or recorded edits. A leftover, swapped, truncated, or otherwise mismatched file can therefore be used with a different model checkpoint. The checkpoint’s config check does not bind individual files to that checkpoint.

Replace the graph payload and load branch with a scheme that stores and checks the full job key, config tag and a stable checkpoint identity; on load, recompute `cr.validate` and require its result to equal the stored edit count. Reject unexpected graph files and fail closed on a missing or mismatched identity. For example, the required load condition should be:

```python
assert d["key"] == k
assert d["config"] == config
assert d["checkpoint_id"] == state["checkpoint_id"]
assert cr.validate(ei, d["g"], n, b) == d["edits"]
```

Generate and atomically save `checkpoint_id` with the model state before generating or importing graph files, and include it in every graph payload. The interruption/resume smoke test should cover interruption during initial import as well as during an attack.

### SHOULD-FIX — “Bit-identical” Jaccard claim needs a bounded qualification and resource check

[followup.py:94](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:94>) forms counts in SciPy float64, converts them to float32, and uses the same float32 degree scatter and final arithmetic as [run.py:119](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/run.py:119>). For the intended simple, unweighted graphs with common-neighbor counts exactly representable in float32, this should match the original; it is not an unconditional bit-identity proof for arbitrary inputs. Also, sparse `adj @ adj` materializes all two-hop pairs, so it removes the dense allocation but does not guarantee a small Pubmed peak.

Replace [protocol.md:32](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:32>) with:

> “The Jaccard score is computed using sparse two-hop counts, converted to float32 before the original degree scatter and division. For the intended simple unweighted graphs, selfcheck requires bit-for-bit equality with the original on the tested Cora graphs. Pubmed peak memory and runtime must be measured in the feasibility gate.”

### SHOULD-FIX — The analysis does not check all own-metric invariants

[analyze_followup.py:80](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/analyze_followup.py:80>) checks only `own_all`. A malformed `own_std` or `own_strong` can pass that assertion and later affect reporting. Replace lines 82–83 with:

```python
for label in ("own_std", "own_strong", "own_all"):
    value = v[f"{label}@{r}"]
    assert (value is not None) == (v["arm"] in ATTACKED), (
        ds, v["arm"], r, label, value
    )
```

The graph count **178** at [analyze_followup.py:25](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/analyze_followup.py:25>) is correct: `2 * (1 + 30 + 1 + 48 + 9)`.

### NIT — Attack-family wording overstates independence

GRBCD is a different greedy attack algorithm, but it is in the same randomized block-coordinate descent family as PRBCD. Replace “an independent attack family” at [protocol.md:57](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:57>) with:

> “a distinct greedy RBCD attack (125 steps, prob-margin).”

The call signatures at [followup.py:243](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:243>) match the published PyG 2.4 and 2.5.1 signatures: PRBCD accepts `lr` and `epochs_resampling`, and GRBCD accepts the supplied arguments. The intended PRBCD settings are 400 epochs, 300 resampling epochs, and learning rates 100, 1,000, and 10,000. The exact cluster PyG version was not established from the repository; verify its signatures at the feasibility gate. [PyG 2.4 documentation](https://pytorch-geometric.readthedocs.io/en/2.4.0/modules/contrib.html), [PyG 2.5.1 documentation](https://pytorch-geometric.readthedocs.io/en/2.5.1/modules/contrib.html).

### NIT — Source header still states the constraint too broadly

Replace [followup.py:3](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:3>) with:

> `Four questions; the added scorer-derived edge weights use structure only. RUNG's existing propagation coefficients also depend on predictions from node features.`

## Earlier findings

| Finding | Status | Assessment |
|---|---|---|
| 1 | **FIXED** | [followup.py:439](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:439>) scores `A_ARMS`. |
| 2 | **FIXED** | The explicit qualification at [protocol.md:15](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:15>) accurately reflects the requested RUNG design. |
| 3 | **PARTLY** | [protocol.md:98](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:98>) appropriately limits inference and frontier claims. It promises all paired cell differences, but analysis reports per-split means and only the masking diagnostic’s individual cells; see [analyze_followup.py:113](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/analyze_followup.py:113>). Add printing of each hypothesis and guardrail’s paired differences. |
| 4 | **PARTLY** | Sparse computation and equality checks exist at [followup.py:88](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:88>) and line 543. Pubmed peak memory remains a required measurement, and the unconditional identity claim needs qualification above. |
| 5 | **PARTLY** | Complete keyed source loading and exact metric assertions exist at [analyze_followup.py:87](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/analyze_followup.py:87>), but the original worst-graph reduction must be restored as above. |
| 6 | **PARTLY** | The matched learning-rate sweep, repeated seeds and greedy attack are declared at [protocol.md:53](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:53>); the claim is appropriately bounded at line 124. Loss trajectories, gradient norms and best discrete accuracy across iterations are still not recorded, as acknowledged at line 176. |
| 7 | **FIXED** | [followup.py:493](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:493>) checks hand-derived derivatives, including zero weight and identity restoration. The displayed derivatives match the implemented normalization for those cases. |
| 8 | **FIXED** | [protocol.md:81](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:81>) correctly describes a sensitivity flag and its limits. |
| 9 | **PARTLY** | Atomic individual graph files and a resource gate are specified at [followup.py:383](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:383>) and [protocol.md:150](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:150>); resume integrity remains blocked above, and measurements remain pending. |
| 10 | **PARTLY** | [followup.py:343](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:343>) verifies old tag, data and code hashes and records old config. Per-file resume provenance and revalidation remain incomplete. |
| 11 | **FIXED** | [followup.py:462](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:462>) returns `None` for empty own sets. Strengthen analysis assertions as above. |
| 12 | **FIXED** | [protocol.md:117](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:117>) includes both comparisons against S-VG in the Holm family and requires them for C4. |
| 13 | **PARTLY** | Short attack settings and a separate integration gate are present, but [followup.py:522](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:522>) still loads real datasets, runs multiple attacks and trains a full pruned VGAE at line 623. It is not the requested tiny synthetic fast check; resume is only planned in the smoke cell. |
| 14 | **FIXED** | [followup.py:207](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:207>) deep-copies; line 234 restores the prior mode and training state. Uniform equivalence is described with tolerance. |

## Earlier NITs

- **Seed assignments — FIXED:** [followup.py:306](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:306>) uses `base + 702`; [protocol.md:28](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:28>) distinguishes initialization from training trajectories.
- **RUNG rescaling and text qualification — FIXED:** [followup.py:195](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:195>) rescales per target; the protocol states the qualification and numerical tolerance.
- **Empty hidden mask — FIXED:** [followup.py:146](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:146>) skips empty or all-hidden draws.
- **Cache CPU assumption — FIXED:** documented at [followup.py:100](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:100>), with CPU conversion for hashing.
- **Graph count — FIXED:** revised count is 178, as checked above.
- **RUNG-VG old-union diagnostic — FIXED:** analysis labels it transfer-only at [analyze_followup.py:145](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/analyze_followup.py:145>).

No experiment, selfcheck, or smoke cell was run for this review. The code and protocol changes above remain to be made; after that, the declared feasibility gate must pass before the full grid.