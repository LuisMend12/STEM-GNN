**Hold the robustness claim until the attack issue is addressed.** The certificate is sound; I found no definite late-run serialization crash. This was a static review: local Python lacks `torch_geometric`, so I could not execute the selfcheck.

Blockers for launch/claims:

- **Tanh masking remains untreated.** `STEM-GNN/scripts/capped/run.py:65`, `STEM-GNN/scripts/llm_stem/run.py:199`, `experiments/capped-correction/protocol.md:60`. All five attacks differentiate through the saturated cap. The STE/soft modes change the VQ backward path only. Reporting a certificate beside measured accuracy does **not** validate empirical improvements above that certificate—especially superiority over MLP-M, whose accuracy necessarily bounds this certificate from above. **Fix:** preregister a cap-specific BPDA attack with the exact capped forward and an identity or nonsaturated backward, add its graphs to the shared union, and retain the qualification that results demonstrate performance against the tested attacks. The current protocol acknowledges masking but does not resolve it.

- **Zero caps can silently poison training and invalidate the certificate.** `STEM-GNN/scripts/capped/run.py:65,115–118`. Validation margins are nonnegative, but a quantile can equal zero; `g / 0` followed by multiplication by zero can produce NaNs. NaN logits also invalidate the mathematical bound. **Fix:** reject nonfinite/nonpositive caps immediately, or preregister `B=0` as an explicit MLP-only branch; fail immediately on nonfinite logits/losses. Do not silently clamp B without documenting the changed grid.

- **“Keeping most of the graph benefit” is untested.** `experiments/capped-correction/protocol.md:7,45,48`. H81 requires only +2 points over MLP-M; it can pass after losing most of S0’s graph gain. **Fix:** either describe the objective as retaining *some* graph benefit, consistent with the formal claim rule, or preregister a clean noninferiority/retained-gain threshold versus S0.

Minor findings and operational fixes:

- **Checkpoint portability, not a demonstrated CLI crash.** `STEM-GNN/scripts/capped/run.py:178,256`. `CapModel` defined in `__main__` can round-trip when the same script is rerun directly. Loading that checkpoint from another entry point can fail because its `__main__` lacks `CapModel`; the in-process selfcheck misses this. **Fix:** save constructor metadata plus state dictionaries, or put classes in a stable importable module and test loading in a fresh process.

- **Checkpoint recovery and provenance are incomplete.** `STEM-GNN/scripts/capped/run.py:145–149,167–178,280`. Nothing is saved until all 72 attacks finish. Hashing omits `code_robustness/run.py`, `stem_fixes/run.py`, encoder/VQ code, data, and dependency versions, so changed dependencies can reuse stale results. **Fix:** atomically checkpoint training and each completed attack; fingerprint these dependencies and record PyG/version/data identifiers.

- **Certification wording is inaccurate.** `experiments/capped-correction/protocol.md:19`: approximately `1−q` of validation nodes satisfy the **margin condition**, not necessarily correct certification; certified accuracy additionally requires MLP correctness. `STEM-GNN/scripts/capped/run.py:4` also incorrectly says an edge edit cannot move a logit by B: deviation from m is at most B, while movement between two graphs can approach **2B**. Fix both descriptions.

- **Finish specifying inference before results.** `experiments/capped-correction/protocol.md:40–48`. Freeze p-value sidedness, null values, CI level, and percentage-point units. State whether thresholds are minimum observed effects or effects established by confidence bounds. The additional S0 comparison is a conjunctive claim gate, so its omission from Holm is not automatically an error, but its inferential treatment should be explicit.

Checks that pass:

- For finite logits and `B>0`, each correction lies in `[-B,B]`; therefore margin `>2B` preserves m’s prediction for every graph. The implemented correct-and-margin test is a valid accuracy lower bound.
- MLP parameters are frozen, and overridden `train()` forces MLP evaluation mode: BatchNorm statistics and dropout remain fixed during finetuning and attacks. The VQ lookup uses frozen parameters/codebook.
- Cap grid, validation-only selection, smaller-B ties, arms, shared union including clean and lower budgets, and Stage 2 seed/training calls match. S0 **clean predictions** and MLP-M should reproduce; S0 union robustness need not match Stage 2 because the attacking arms differ.
- No test-label leakage into fitting or CAP selection found. Test labels enter the explicitly declared evaluation attacker; `y.max()` supplies class count.
- Weighted forward and `attack_mode` delegation are correctly wired. [PyG’s PRBCD source](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/contrib/nn/models/rbcd_attack.html) confirms the positional `model(x, ei, w)` call.
- JSON tuples are safe here: graph keys are converted to strings, and tuple values serialize as arrays.