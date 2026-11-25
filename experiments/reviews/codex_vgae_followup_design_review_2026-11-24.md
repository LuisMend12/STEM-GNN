## Findings

**NO-GO before fixes.** The Cora adaptive run produces an extra arm that the analysis rejects, the Pubmed Jaccard implementation has a serious memory risk, and the current inference and “frontier” claims exceed what the design can establish. The attack suite is a useful strengthening, but it does not yet rule out attack weakness.

### BLOCKER

1. **Cora Part A scores an undeclared arm, so analysis fails.** [followup.py:360](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:360>) loads all original models, including `S-VGr`; [followup.py:384](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:384>) scores all of them. [analyze_followup.py:62](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/analyze_followup.py:62>) requires exactly the nine declared arms.

   Replace lines 384–385 with:

   ```python
   rows = [score_arm(arm, models[arm], meta, graphs, kinds, set(old_keys),
                     x, ei, y, te, split, seed)
           for arm in A_ARMS]
   ```

2. **The standing text constraint needs an explicit decision for RUNG.** [followup.py:183](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:183>) supplies structure-only VGAE input weights, but [models.py:140](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/adaptive_robustness/models.py:140>) computes additional propagation coefficients from predictions derived from `x`. Thus RUNG’s *effective* edge coefficients depend on text-derived features. If “edge weights must never be computed from text” includes these coefficients, both RUNG arms violate the constraint; a structure-only RUNG propagation rule is needed. If the intended constraint applies only to the added VGAE scorer, explicitly narrow it in the protocol before running:

   > “The added VGAE and Jaccard input edge weights use graph structure only. RUNG-lite’s existing prediction-dependent propagation coefficients use node features; they are outside this narrower constraint.”

   That narrower wording requires acceptance of the constraint interpretation; the present protocol’s unqualified statement is inaccurate.

3. **The inferential unit is overstated, especially for Pubmed.** [protocol.md:75](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:75>) treats 15 or nine correlated split/seed cells as independent paired-t observations. Seeds share the same graph and test population; three split effects are repeatedly sampled. Part R also repeats only three splits. The stated caveat at line 80 does not make the resulting nominal confidence intervals or Holm p-values valid for independent graph or dataset replication.

   Replace lines 75–80 with:

   > “The planned cell-level paired summaries, t intervals, and Holm-adjusted p-values are descriptive sensitivity analyses conditional on these fixed graphs and splits. Cells are dependent and are not independent dataset replications. Report all paired cell differences and split-level means. Do not use these p-values or intervals to make confirmatory population-level claims.”

   Replace lines 96–102 with:

   > “Report Cora and Pubmed results separately as conditional evidence on two fixed graphs. A gain on both is a replication signal, not sufficient by itself to establish a new frontier. Establishing a frontier requires independent datasets or graph instances, stronger independently validated attacks, and comparisons with competitive tuned baselines.”

4. **Pubmed can exhaust memory in the Jaccard baseline.** [run.py:121](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/run.py:121>) allocates an `n × n` float matrix, and line 123 produces another dense multiplication result. At 19,717 nodes, each float32 matrix alone is about 1.45 GiB; training and repeated evaluation can require several GiB more. Caching at [followup.py:91](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:91>) avoids recomputation on an identical graph but does not remove this peak.

   Replace the dense `JaccardScorer.scores` implementation with an exact sparse neighbor-intersection implementation that returns scores in the original edge order. Add a small-graph equality test against the existing implementation and a Pubmed memory/time smoke test before committing the full grid. Do not silently drop S-SJ or approximate its values: either changes the declared comparison.

5. **Reproduction is not enforced as declared.** [analyze_followup.py:83](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/analyze_followup.py:83>) allows an empty or incomplete set of old A JSONs to pass with `worst = 0`; lines 94–101 only print R differences, despite the protocol requiring reproduction. It also checks only clean and old worst-node for A, not all original reported metrics.

   Replace the glob-based loops with keyed loading that asserts the complete expected 15 A and 50 R source cells, the original tag, and expected shared arms. Assert exact equality for the declared shared metrics, including R accuracy and macro-F1, and A clean, worst-node, worst-graph, own worst-node, and transfer accuracy. Fail with the cell, arm, metric, old value, and new value. Add the needed old-union A metrics to `score_arm` first.

### SHOULD-FIX

6. **Attack effort does not yet establish that the adaptive gain survives a sufficiently strong attack.** [protocol.md:43](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:43>) increases epochs and block size but leaves the optimizer’s default step size unchanged. More epochs can still leave VGAE gradients ineffective; the three “restarts” also change loss or backward mode rather than supplying multiple seeds for each configuration. Add a predeclared learning-rate sweep for the **true differentiable VGAE path**, including substantially larger and smaller rates, multiple seeds for the strongest configurations, and an alternative objective targeted to the endpoint, such as a margin or per-node targeted loss. Record loss trajectories, gradient norms, edit counts, and best discrete accuracy across iterations. Include an independent attack family or transfer from independently trained surrogates. Select by attack loss or validation procedure declared before test scoring, and retain all resulting valid graphs in the union.

   Replace the claim at [protocol.md:97](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:97>) with:

   > “C1: S-VG retains a gain on the declared expanded attack union; this is evidence against the attacks in that suite, not evidence that gradient masking or stronger attacks are absent.”

   The shared union gives both arms the same evaluation graphs and is a fair comparison on that union. However, three extra weighted-arm attacks versus three extra unweighted-arm attacks do not by themselves establish matched optimization effort or equal attack strength. Apply the learning-rate and restart additions to S0 as well. Avoid presenting an unmatched number of attack attempts as equivalent effort.

7. **`bpda_flat` and `detach` are less distinct than their names suggest.** [followup.py:202](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:202>) correctly preserves true forward values and substitutes the normalized-uniform derivative; `detach` preserves the true normalization derivative while detaching scorer scores. Both suppress the scorer gradient. They do **not** test a larger or differently scaled true VGAE gradient. The monkeypatch reaches all three weighted classes because they resolve `vw.neighbor_weights` at call time, and `finally` restores it. Add explicit checks of the expected derivative against a hand-computed small graph, including zero edge weights, and verify restoration by identity, not merely function name.

8. **The masking diagnostic is a one-sided flag, not a test for masking.** [protocol.md:61](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:61>) implies that a drop of three points identifies obstruction in the original attacks. It could also reflect extra optimization or transfer; no drop cannot exclude masking.

   Replace lines 61–63 with:

   > “Descriptively report the paired change in S-VG own-attack worst-node accuracy when the weight-aware graphs are added, including individual cells. A drop of at least three points flags sensitivity to the added attacks; it does not identify the cause. No drop does not rule out gradient masking.”

9. **Checkpointing and Pubmed compute need an explicit feasibility gate.** [followup.py:371](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:371>) and line 381 rewrite all models and accumulated graphs after every attack. The planned count is correctly **116 graphs per cell**: `2 × (1 + 6 × 8 + 9)`. Pubmed also scores each graph for every arm, with VGAE encoding and, currently, dense Jaccard. Peak PRBCD memory can be dominated by 200,000 candidate edges and per-edge activations, especially for STEM and RUNG.

   Replace [protocol.md:112–117](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:112>) with:

   > “Before the full grid, measure peak memory, attack time by arm and attack type, scoring time, and checkpoint size and write time on a Pubmed smoke cell. Verify graph validity and resume after interruption. Save attack graphs atomically as individual files and save model state separately. If resources require changing the suite, preregister a single dataset-wide amendment before inspecting accuracy; report the amended suite and do not describe it as the original strong settings.”

   The current cost rule says the same settings for every arm even though the suite intentionally differs by arm. Specify precisely which settings change and preserve the weight-aware components.

10. **Resume provenance is incomplete.** [followup.py:350](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:350>) checks configuration equality and job keys correctly, and the set assertion catches unexpected keys. But old Cora checkpoints are validated only by their stored config hash, not by comparison with the current data and dependency hashes; an old checkpoint can be applied to changed inputs. Verify the old checkpoint’s recorded data hash and relevant code/environment identity against the actual original artifacts, and record both old and new provenance in the output. Validate each graph immediately after generation and before checkpointing, rather than only after the whole suite.

11. **Unattacked arms get misleading “own” metrics.** [followup.py:399](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:399>) gives unattacked arms clean accuracy for `own_*`, because the own graph list is empty. Replace lines 404–406 with:

   ```python
   for label, selected in (
       ("own_std", [k for k in own if kinds[k] == "std"]),
       ("own_strong", [k for k in own if kinds[k] in ("std", "strong")]),
       ("own_all", own),
   ):
       row[f"{label}@{r}"] = worst(selected) if selected else None
   ```

   `old_worst_node` is correctly limited to old Cora graph keys; it is `None` on Pubmed. The old union for RUNG-VG is a legitimate transfer diagnostic, but it cannot reproduce an old RUNG-VG result.

12. **The noise-aware hypothesis does not isolate improvement over the original VGAE.** [protocol.md:86](</G:/broswer download/STEM-GNN/experiments/vgae-followup/protocol.md:86>) compares both variants only with S0. A positive result could reflect weighting generally, not a fix to the original VGAE’s null result.

   Replace C4 at line 100 with:

   > “C4: A noise-aware variant improves accuracy over S0 under the declared injection level. Separately report its paired difference from S-VG; improvement over S-VG is required to describe the noise-aware modification as an improvement over the original VGAE.”

   If the latter is intended as a confirmatory claim, add both variant-versus-S-VG comparisons to the preregistered multiplicity family before running.

13. **The self-check is not lightweight.** [followup.py:525](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:525>) runs multiple 400-epoch, 200,000-block attacks; line 513 also trains a full VGAE. Split it into a fast correctness check using tiny synthetic graphs and short attack settings, and an explicitly named integration smoke test. Include a resume/interruption test and tests that `twin` leaves the original model and its mode unchanged.

14. **`twin` is a shallow copy and shares modules.** [followup.py:192](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:192>) gives the copy its own scalar `tau`, so that assignment does not leak. But its underlying modules are shared, and `eval()` and `attack_mode` mutation affect shared state. Current attack cleanup resets the mode to `"ste"`, not necessarily its prior value.

   Replace lines 192–194 with:

   ```python
   t = copy.deepcopy(model)
   t.tau = 1e6
   return t
   ```

   In `prbcd_strong`, save the previous mode and training state and restore both in `finally`. `1e6` is an approximation to uniform scores, so keep tolerance-based equivalence tests rather than claiming exact equality.

### NIT / verified points

- [followup.py:240](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:240>) matches the shared `train_all` seed assignments, including `base + 702` for both RUNG fits. Equal seeds align initialization; subsequent training trajectories and selected epochs can differ.
- [followup.py:178](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:178>) rescales normalized incoming weights by incoming total. Uniform scores reproduce RUNG’s input weights, subject to numerical tolerance. The VGAE and Jaccard scorers inspected do not receive `x`; the RUNG qualification above concerns its separate internal coefficients.
- Masking at [followup.py:131](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/followup.py:131>) hides undirected pairs together and samples negatives against the full graph. Pruning at line 150 keeps symmetric pairs. Handle an empty hidden set defensively for small graphs; it is unlikely on these datasets.
- The Jaccard cache hashes the entire ordered edge tensor, so different orderings safely miss the cache rather than return misaligned scores. It assumes CPU tensors; document that assumption or move to CPU explicitly for hashing.
- The analysis graph-count assertion is correct. The old Cora checkpoint has 52 graphs; adding the new jobs yields 116 per cell.
- The check at [analyze_followup.py:130](</G:/broswer download/STEM-GNN/STEM-GNN/scripts/vgae_weighting/analyze_followup.py:130>) suppresses RUNG-VG’s available old-union diagnostic. Consider displaying it as transfer-only.

**NO-GO.** Resolve the Cora arm mismatch, the interpretation of the text constraint, sparse Pubmed Jaccard, reproduction enforcement, and the statistical claims before running. Then perform only the declared lightweight and resource smoke checks before launching the full experiment. No experiment or heavy computation was run for this review.