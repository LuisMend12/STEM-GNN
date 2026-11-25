No blocker invalidating H75 found. The analysis implements Amendment 2’s comparisons, support/against rules, separate Holm families, clean guardrail, and staged tests correctly. Regenerated output matches the saved output exactly; all reported numerical results match, including rounding.

Fix before finalizing the report:

- **Overstated robustness conclusion — `experiments/llm-stem/analysis_stage2.md:7`.** “No Qwen placement improves” treats inconclusive H72 as a negative finding. Replace with “No tested placement met preregistered robustness support; S-A was inconclusive.”

- **Missing low-label diagnostic data — `STEM-GNN/scripts/llm_stem/stage2.py:331`, `experiments/llm-stem/analysis_stage2.md:32`.** Saved per-node results contain full-label predictions only. Downloading those files cannot establish the proposed low-label teacher diagnostic. Save low-label predictions/correctness and regenerate them deterministically from checkpointed encoders. Also replace “would mean … falling back” with “would be consistent with greater reliance on Qwen”; concentration alone does not establish that mechanism.

Minor issues:

- **Unsupported validation-budget comparison — `experiments/llm-stem/analysis_stage2.md:26`.** Nothing establishes that 500 validation labels favor MIX and S0 *more than S-A*. S-A also uses calibration and checkpoint selection; fixed λ does not establish relative benefit. Delete that comparison and retain the untested-budget caveat.

- **Overbroad leakage wording — `experiments/llm-stem/analysis_stage2.md:23`.** “Their labels aren’t [used]” omits the declared pooled audit, where another split’s validation nodes can be current test nodes (`protocol.md:199`). Say “Current-split test labels do not enter low-label training or calibration; the frozen cross-split audit has the declared overlap.”

- **Silent result replacement — `STEM-GNN/scripts/llm_stem/analyze_stage2.py:25`.** Multiple config files for one cell silently overwrite each other. Reject duplicate keys and assert a common configuration plus the exact split/seed grid. Current files are consistent, so this does not affect these results.

The low-label computation is sound on inspection: S-A retrains with 21 training labels and KL outside low-train/validation; MIX uses the retrained low-label S0; both MLPs retrain on the reduced mask; QWEN retains its validation-calibrated prior. Test-label-informed attack graphs do not feed into low-label training or `low_clean`.