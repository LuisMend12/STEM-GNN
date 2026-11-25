I reran `analyze.py` against all 15 JSONs. Its numerical output reproduces `analysis_output.txt`. The reported means, CIs, wins, split effects, retained gain, and table entries in `analysis.md` match at their stated precision. CAP* exactly copies the selected arm in every cell, and independently recomputing selection picks CAP0.1 in all 15.

1. **BLOCKER — Baseline checkpoint serialization will fail.**  
   `STEM-GNN/scripts/capped/baselines.py:40–42,171` dynamically imports classes as `ar_models` without registering that module in `sys.modules`. The first checkpoint save includes instances of those classes; pickle cannot resolve their module. The selfcheck never tests baseline checkpoint serialization.  
   **Fix:** insert `sys.modules["ar_models"] = arm_models` before `exec_module`, and add a checkpoint round trip in a fresh process that imports the baseline entry point.

2. **SHOULD-FIX — The proposed frontier criterion does not establish a better frontier.**  
   `experiments/capped-correction/baselines_protocol.md:54–56` counts dominating *some* ENS point. That ENS point could itself be dominated by another baseline, while CAP remains below the baseline frontier. “Similar clean accuracy” also has no numerical definition.  
   **Fix:** compare CAP against the nondominated baseline envelope at prespecified clean-accuracy targets or tolerances. Distinguish “one CAP point improves the baseline envelope” from superiority across the frontier.

3. **SHOULD-FIX — Exploratory dominance is inconsistent with the proposed success rule.**  
   `experiments/capped-correction/baselines_protocol.md:49–55`; `STEM-GNN/scripts/capped/analyze_baselines.py:74–80`. The code examines 54 arm pairs, with two endpoints each, and labels pairs using unadjusted intervals. The protocol then uses finding some dominance as the answer to the central question. Searching those labels is not confirmatory evidence.  
   **Fix:** retain these as exploratory results, or prespecify and multiplicity-correct the frontier success test. For a fixed pair, requiring both endpoints positive is an intersection-union test; the main multiplicity problem is searching across pairs.

4. **SHOULD-FIX — Family B answers robustness superiority, not frontier superiority.**  
   `experiments/capped-correction/baselines_protocol.md:44–47`. CAP* was selected by the original certificate-weighted rule and is near MLP-M. Beating a cleaner baseline on worst-node accuracy can simply reflect choosing less graph influence. Holm over the eight comparisons is implemented correctly, but does not resolve this interpretation.  
   **Fix:** explicitly restrict Family B’s conclusion to worst-node differences for the fixed selected arm. Use a separate matched-clean criterion for frontier conclusions.

5. **SHOULD-FIX — ENS is useful, but incompletely matched.**  
   `STEM-GNN/scripts/capped/baselines.py:71–72`; `experiments/capped-correction/baselines_protocol.md:28–32`. Log-probability mixing is equivalent, for predictions, to weighted logit mixing: the log-normalization terms are class-independent. Its effective balance therefore depends on the two models’ logit scales. Three weights may miss CAP’s clean-accuracy range. Moreover, CAP trains a new graph correction through the combined predictor, while ENS reuses independently trained S0.  
   **Fix:** describe ENS as a shared-component, untrained control. Prespecify a denser validation-selected weight grid and, if intended, validation-only temperature calibration. RES remains the closer control for the effect of capping during joint training.

6. **SHOULD-FIX — Resume checks do not establish source/data compatibility.**  
   `STEM-GNN/scripts/capped/baselines.py:148–161,254–259`. Resume checks only the source checkpoint’s basename. The baseline checkpoint does not bind the current data, masks, runtime configuration, or source contents. A changed data file or replaced source checkpoint can silently mix saved models/graphs with new inputs.  
   **Fix:** store and compare source-content and data hashes, split/seed, rates, threads, and relevant environment versions. Save the initialized baseline checkpoint before starting attacks as well.

7. **SHOULD-FIX — Reproduction checks are too weak to protect the actual run.**  
   `STEM-GNN/scripts/capped/baselines.py:186–198`; `STEM-GNN/scripts/capped/analyze_baselines.py:45–46`. The selfcheck tests only split 0/seed 0, before adding baseline graphs. Analysis merely prints drift, so it proceeds even if reproduction fails. It also never asserts the promised extended-union monotonicity.  
   **Fix:** for every cell, assert original-union reproduction for all original arms at both budgets, unchanged clean/certified accuracy, and extended worst-node ≤ original worst-node. Exercise scoring with added baseline graphs in the selfcheck.

8. **SHOULD-FIX — `orig_worst_node` uses exclusion instead of source membership.**  
   `STEM-GNN/scripts/capped/baselines.py:127–129`. Excluding keys whose owner is in `BASELINES` works with today’s names, but does not encode “graphs from the original checkpoint.” A future baseline omitted from that tuple would contaminate the reproduction metric.  
   **Fix:** capture the original graph-key set immediately after loading and pass it explicitly to `score`. Current graph names do not collide, and the merge-length assertion correctly catches collisions.

9. **SHOULD-FIX — Degenerate paired differences can corrupt Holm results.**  
   `STEM-GNN/scripts/capped/analyze.py:33,37–42`, reused by baseline analysis. Zero standard error produces `NaN`; `holm` can then convert that invalid value into a misleading adjusted value through its `min`/`max` operations. This does **not** affect the observed H80–H82 results.  
   **Fix:** explicitly handle all-zero differences, define or reject constant nonzero differences, and reject nonfinite p-values before Holm adjustment.

10. **SHOULD-FIX — The independence limitation needs stating.**  
    `experiments/capped-correction/analysis.md:21`; `experiments/capped-correction/baselines_protocol.md:44`. Fifteen training cells are not fifteen independent datasets: they share Cora, with five seeds per split and overlapping graph/test information. The calculations follow the prescribed paired-t procedure, but their uncertainty should not imply dataset-level replication.  
    **Fix:** state this limitation explicitly and report split-level sensitivity alongside the prescribed analysis. Broader claims require additional datasets.

11. **SHOULD-FIX — The causal interpretation overstates H82.**  
    `experiments/capped-correction/analysis.md:54–55,69–70`. RES already improves worst-node accuracy by 8.06 points. H82 supports an additional gain for selected capped correction over RES under this suite; it does not establish that residual design contributes nothing.  
    **Fix:** write: “Residual correction improves robustness descriptively; selected capped correction improves it substantially further, with lower clean accuracy.”

12. **SHOULD-FIX — Selection explanation uses test summaries and asserts an untested alternative.**  
    `experiments/capped-correction/analysis.md:61–65`. Selection actually uses per-cell validation scores, while the explanation cites test certification. “A rule weighted toward clean accuracy would pick a middle cap” needs a specified weight and validation recomputation; sufficiently strong clean weighting could select a large cap.  
    **Fix:** show validation selection scores and gaps. Remove the middle-cap assertion or demonstrate it for an explicitly descriptive alternative weight.

13. **NIT — Tightness and dominance wording needs precision.**  
    `experiments/capped-correction/analysis.md:30–31,49–53`. CAP* is worse on robustness and better clean, so “dominated … under attack” should simply say “lower worst-node accuracy.” The 2.35-point slack does bound the remaining gap between certified accuracy and empirical worst-node robustness, but does not prove PRBCD is near-optimal for a single shared graph. Nor does a large slack establish that the certificate is attainable.  
    **Fix:** identify the bound as one on the nodewise robustness metric, and say stronger attacks *may* lower the larger-cap results.

14. **NIT — H81’s explanation should name the upper CI explicitly.**  
    `experiments/capped-correction/analysis.md:26`. The verdict is correct: mean 1.03 < 2 prevents support; upper CI 1.36 ≥ 1 prevents “against.”  
    **Fix:** replace “above the +1 against threshold” with “upper CI is not below +1.”

15. **NIT — Repeated original-result loading is unnecessary.**  
    `STEM-GNN/scripts/capped/analyze_baselines.py:45,83–86`. `an_orig` reloads all original JSONs for each arm. This is redundant, not a current numerical bug.  
    **Fix:** load originals once; also assert expected arm/cell membership and finite metrics explicitly.

**Part 1 verdict:** Numerically reproducible and protocol-correct. H80 is **AGAINST**, H81 **INCONCLUSIVE**, H82 **SUPPORTED**, the gate **MET**, and the claim **NOT MET**. Holm, certificates, cumulative shared unions, and CAP* construction show no current number-changing bug. Revise the interpretation and caveats. Slurm success and the historical selfcheck pass cannot be independently established from these JSONs alone.

**Part 2 verdict:** **Do not launch unchanged:** fix checkpoint pickling first. The splits, baseline training masks, cumulative shared union, and LogMix attack-mode forwarding/reset are coherent. Resume reconstruction of ENS also looks correct. Strengthen provenance and reproduction assertions, and revise the frontier success rule before treating this as a confirmatory answer about CAP’s frontier.