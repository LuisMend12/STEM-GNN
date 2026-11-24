The original claim-level blockers are resolved. The checkpoint-class fix also works in an isolated pickle test, including when the controls runner is `__main__`. Some integrity and selfcheck weaknesses remain.

This is a static review plus isolated checks: the local directory contains all 15 Stage 2 result JSONs, but **no Stage 2 checkpoints**, so I could not verify real checkpoint loading or numerical reproduction.

File references below use:
- **L:** `STEM-GNN/scripts/llm_stem/lowlabel_controls.py`
- **A:** `STEM-GNN/scripts/llm_stem/analyze_lowlabel_controls.py`
- **S:** `STEM-GNN/scripts/llm_stem/stage2.py`
- **P:** `experiments/llm-stem/lowlabel_controls_protocol.md`

| Original finding | Status | Evidence |
|---|---|---|
| 1. `__main__` checkpoint classes | **FIXED** | `L:35–38` registers modules and injects all three Stage 2 classes into the active `__main__`. Stage 2’s only class definitions are `Mix` (`S:173`), `MLP` (`S:204`), and `PriorOnly` (`S:213`). Actual checkpoint loading remains unverified. |
| 2. Unqualified realistic-validation claim | **FIXED** | Identity-only S-A0 is implemented at `L:205–206`; strict-budget claims use it at `P:42–45,81–85`. Historical S-A audit dependence and the 30-node audit are disclosed at `P:118–119`. |
| 3. V21 LLM-specificity claim rule | **FIXED** | H94 compares S-A0 against S-T under V21 (`A:62`). The protocol separates strict-budget gains from superiority over this teacher and avoids interpreting nonsignificance as equivalence (`P:81–85`). |
| 4. Uniform control misdescribed | **FIXED** | `P:54` correctly identifies KL(U‖p), distinguishes it from the entropy confidence penalty, and disclaims entropy matching. `P:79–80` narrows the inference. |
| 5. Artifact identity | **PARTIAL** | Explicit Stage 2 tag and config checks: `L:65–80`; audit validation: `L:194–206`; analysis identity set: `A:20,28,37`. However, checkpoint contents and prior-logit contents are not fingerprinted. See below. |
| 6. Conditional interpretation of 15-cell tests | **FIXED** | Shared graph/split dependence is explicit at `P:64–66`; per-split mean differences are reported at `A:102–107`. These are descriptive sensitivity summaries, not a dependence-adjusted inferential analysis. |
| 7. Teacher interpretation and diagnostics | **PARTIAL** | Comparator wording and validation reuse are narrowed/disclosed (`P:55,120–123`). Teacher best epoch and validation NLL are still absent: `L:89–95,110–113`; `S:222–227` discards the fit return values. |
| 8. Selfcheck rejects null effects | **PARTIAL** | Argmax-change requirement is removed, but `L:177–181` still requires every fitted treatment to change logits beyond `allclose` tolerance. This remains an outcome-dependent launch gate. |
| 9. Repeated selfcheck computation | **PARTIAL** | Predictions are now computed once (`L:182`), and V21 has two full fits instead of three (`L:177,183–184`). Teacher fitting is still repeated separately for distribution checks, scrambling checks, and those full fits (`L:164–175`). |
| 10. Empty diagnostic subsets | **FIXED** | `A:91–100` pools integer numerators/denominators and explicitly prints `empty` for a zero denominator. It no longer averages cell-level NaNs. |

**Targeted checks**

- **`__main__` injection works.** I executed the actual injection loop against synthetic objects pickled as `__main__.MLP`, `__main__.Mix`, and `__main__.PriorOnly`; all resolved to their replacement `s2` classes. There are no additional class definitions anywhere in the current `stage2.py` AST. Other saved model classes come from imported modules rather than Stage 2’s `__main__`.
- **`STAGE2_CONFIG` is correct.** Its values match the relevant `env` fields in **all 15** supplied Stage 2 JSONs. Recomputing Stage 2’s configuration tag gives `ce407cf8cd`; the audit bytes also match the specified audit hash. Stage 2 constructs `arms` with `sorted(args.arms)` (`S:473–475`), and the constant has exactly that order. Dictionary key order is irrelevant to `==`; list order remains significant, correctly. `L:79` requires exact checkpoint-config equality and equality of the corresponding JSON environment fields.
- **`load_priors` preserves Stage 2’s selection.** `L:199–204` checks the audit rule, text hash, frozen audit hash, and `[0,2,4]` selection. S-A loads those rotations; S-A0 loads `[0]` (`L:205–206`). It does not accidentally replace historical S-A with identity-only logits.
- **V500 still matches the intended Stage 2 recipe.** Full `va` reaches calibration and checkpoint selection (`L:127–130`, `L:87–91`); S-A uses `[0,2,4]`; low-label masks, finetune seeds, encoder reuse, and anchor exclusions agree with `S:230–254,294,310–311`. MLP-M retains `base+13` (`S:223`). This establishes recipe equivalence, not measured reproduction.
- **H90–H94 and claim logic are consistent.** `A:58–64` puts exactly five tests into one Holm family. Percentage-point conversion, paired tests, and Holm come from `scripts/capped/analyze.py:29–49`. `A:67–73` implements the preregistered point-estimate/p-value thresholds, Claim B’s H92-and-H93 conjunction, and separate teacher comparisons.
- **Pooled teacher diagnostics are arithmetically correct.** `A:93–100` conditions on teacher correctness within each cell’s test mask and sums student-correct counts. The denominators count **node-cell observations**, including repeated nodes across seeds/splits; they are not unique-node counts.

**Remaining issues**

1. **SHOULD-FIX — Artifact identity still permits silent content changes.**  
   `L:230–232` records a Stage 2 tag, audit hash, and text hash, but no checkpoint or prior-logit fingerprint. `S:67–71` validates prior text metadata and NaNs, not content identity. Replacing prior logits while retaining metadata, or replacing checkpoint tensors while retaining config, passes these checks and the identity set at `A:20`. Record prior content hashes shared across cells and checkpoint hashes keyed by split/seed; validate against a frozen manifest. Different cells legitimately have different checkpoint hashes.

2. **SHOULD-FIX — Selfcheck still gates launch on treatment separation.**  
   `L:181` can reject valid weak/null effects. Separately, `L:163` requires S-A and S-A0 priors to differ, although identical logits would not inherently indicate broken loading. Check prescribed inputs and finite, nonzero KL gradients on a controlled fixture; report fitted-logit differences descriptively.

3. **SHOULD-FIX — Prior validation accepts infinities and lacks explicit shape checks.**  
   `load_priors` delegates to `S:63–73`, where only NaNs are rejected. Infinite logits can produce NaN calibrated targets; wrong dimensions can fail later during fitting. Validate expected node/class dimensions and `torch.isfinite(...).all()` before calibration.

4. **SHOULD-FIX — Analysis does not bind prediction files to reported accuracies.**  
   `A:35–36` trusts each paired `_preds.pt`; `A:94–98` then uses it independently of JSON accuracy rows. A stale prediction file can yield correct hypothesis tables but incorrect teacher diagnostics. Validate prediction shapes, labels/test masks, complete arm/regime sets, and reconstructed test accuracies against JSON rows.

5. **NIT — Clarify pooled denominators and persist teacher diagnostics.**  
   Label `n` as node-cell observations at `A:88–100`. Saving teacher best epoch and validation NLL would also finish original finding 7.

I found **no new BLOCKER**. The revised design supports the narrower preregistered claims; an actual checkpoint load and cluster selfcheck remain required before the array launch.

**GO**