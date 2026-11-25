Read-only review completed; no files changed or training run. **Blockers remain despite the passing selfcheck.**

Earlier findings, in order:

1. **Predictor width — fixed.** `make_predictor()` derives the actual projected width, preserving VQ architecture.
2. **Audit masks — mostly fixed.** Correctly stacks splits 0–2 and requires a nonempty common scored subset. Still checks NaN rather than all nonfinite values.
3. **Audit selection/provenance — partial.** Jobs consume the manifest, and 56.7% − 43.3% correctly selects `{0,2,4}`. Manifest validation and immutable provenance remain missing.
4. **RNG isolation — partial.** Torch streams are isolated, but Python RNG is not; see below.
5. **MIX attack strength — resolved for the narrowed claim.** Full-strength S0 candidates are shared, and conclusions explicitly concern the finite suite. Attack adequacy is not established.
6. **Low-label reporting — mostly fixed.** `low_alpha`, validation scores, STEM checkpoint epochs and selected node IDs are recorded. MLP checkpoint epochs and full audit provenance remain absent.
7. **Selfcheck — substantially fixed.** Assertions and direct gradients address the central defects. Its margin uses the maximum including the true class, so it does not exactly reproduce PRBCD’s strongest-wrong-class margin.
8. **Statistics — mostly fixed.** Clean guardrail, staged Holm family and oracle interpretation are clarified. Explicit test direction/nulls for H70–H75 remain unspecified.
9. **Recovery — incomplete and currently broken.** See blockers below; resource requirements also remain unverified.

**New blockers in the changed code:**

- **BLOCKER — checkpoint serialization** (`stage2.py:289`). Saving whole models pickles `ls.RouterCodeModel`, but dynamically loaded `ls` is neither registered in `sys.modules` nor normally importable. The first checkpoint save will fail after the attacks. Save state dictionaries with reconstruction metadata.
- **BLOCKER — unsafe resume/configuration identity** (`stage2.py:278,437`). Checkpoint names omit attack arms, prior/audit identity and code/configuration hashes; loaded checkpoints are not validated. A restricted-arm run can silently supply an incomplete union to a later full run, or old models can be combined with a newly calibrated prior and misleading metadata.
- **BLOCKER for the claimed RNG isolation** (`stage2.py:79,115`). `seeded()` saves/seeds only Torch, whereas [PyG negative sampling uses Python `random.sample`](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/_negative_sampling.html). Its advertised per-epoch seed therefore does not control negative samples.
- **BLOCKER for frozen audit provenance** (`stage2.py:335,430`). `--audit` overwrites the manifest; consumers read only `prior_perms`, without validating hashes, evidence, rule or allowed selection. Outputs retain only the permutation list, so the decision used cannot be authenticated.

Recovery also saves nothing until **all attacks finish**, and does not checkpoint low-label models. No additional blocker found in `make_predictor`, `train_all` metadata or scalar row construction.