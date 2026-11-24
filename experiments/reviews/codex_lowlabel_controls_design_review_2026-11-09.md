**NO-GO as written.** One likely cluster crash and two claim-level problems need fixing before launch. This is a static review; cluster artifacts were unavailable.

Paths below are relative to the repository root.

1. **BLOCKER — Loading the whole Stage 2 checkpoint likely fails on `__main__` classes.**  
   `STEM-GNN/scripts/llm_stem/lowlabel_controls.py:35,108`  
   Stage 2’s documented direct invocation saves its `MLP`, `Mix`, and `PriorOnly` instances as `__main__.MLP`, etc. Registering `s2` does not resolve those names. `torch.load` unpickles the **entire checkpoint** before indexing `["encoders"]["S0"]`, so unused saved models still matter. Expect `AttributeError: Can't get attribute 'MLP' ...` or equivalent.  
   **Fix:** use a controlled unpickler mapping those Stage 2 `__main__` classes to `s2`, or export an encoder-only checkpoint through the original Stage 2 entry point. Test against an actual saved checkpoint before submitting the array.

2. **BLOCKER — V21 cannot support the unqualified “realistic validation” claim.**  
   `experiments/llm-stem/lowlabel_controls_protocol.md:37,66,96`  
   The frozen audit selected rotations using labels outside V21, including nodes that can be test nodes in another split. Freezing the choice prevents further adaptation; it does not remove that label dependence. “Their text is used, never their labels” is therefore false for the complete pipeline. Also, the audit used **30 sampled validation nodes**, not all 500 per split; the limitation should state the actual selection procedure.  
   **Fix:** retain the historical prior for reproduction, but add a preregistered identity-rotation arm for the strict V21 claim, or explicitly restrict the claim to “21 validation labels for downstream fitting, conditional on the previously selected prior.” State the total supervised budget as **21 training + 21 validation labels**.

3. **BLOCKER — The claim rule does not establish LLM specificity under V21.**  
   `experiments/llm-stem/lowlabel_controls_protocol.md:57,64,69`  
   H91 tests S-A versus S-T only under V500. It cannot decide whether an advantage under V21 is specific to Qwen rather than this text teacher. Conversely, failing H91 does not establish “just text-teacher distillation”: failure to show superiority is not equivalence.  
   **Fix:** add S-A − S-T under V21 to the Holm family and require it for the V21 LLM-superiority claim. Use “superiority over the specified MiniLM MLP was not established” for an inconclusive result.

4. **SHOULD-FIX — S-U is a valid uniform-target control, but it is not the stated confidence penalty.**  
   `experiments/llm-stem/lowlabel_controls_protocol.md:46`; `STEM-GNN/scripts/llm_stem/stage2.py:157`  
   The implementation minimizes `KL(U || p) = −mean_c log p_c − log C`. The usual entropy confidence penalty corresponds to `KL(p || U) = log C − H(p)`. These have different gradients. Keeping λ = 1 fairly holds the implemented distillation coefficient fixed, but does not match regularization strength or target entropy. Beating uniform alone also cannot isolate semantic content from confidence differences.  
   **Fix:** describe S-U as “uniform-target cross-entropy regularization.” Narrow the inference to that control; use an entropy-preserving shuffled Qwen target if claiming node-specific content. A training-label-marginal arm would duplicate uniform because the training mask is class-balanced.

5. **SHOULD-FIX — Artifact identity is insufficiently checked and can silently mix experiments.**  
   `STEM-GNN/scripts/llm_stem/lowlabel_controls.py:57,64,189`; `STEM-GNN/scripts/llm_stem/analyze_lowlabel_controls.py:27`  
   Checkpoint and reference JSON are selected independently by glob. Their configurations are never matched. Unlike Stage 2’s entry point, this runner does not validate the audit’s text hash/rule or record its hash. Analysis checks only the source hash, allowing cells with different audit files, priors, or environments into one family.  
   **Fix:** select one explicit Stage 2 tag, validate checkpoint and JSON configuration equality, restore Stage 2’s audit checks, and record/validate checkpoint, audit, and prior identities across cells. Fail before training on mismatches.

6. **SHOULD-FIX — The 15-cell t-test needs its conditional interpretation restored.**  
   `experiments/llm-stem/lowlabel_controls_protocol.md:52`  
   Pairing is implemented correctly, but five seeds share each split and all cells share Cora. Holm corrects multiplicity, not dependence. The previous analysis explicitly acknowledged this; the new protocol drops that qualification.  
   **Fix:** label these as run-level, graph-conditional intervals and add a split-level sensitivity summary. Do not present the 15 cells as independent dataset replications.

7. **SHOULD-FIX — S-T is fair as a specified recipe, but a weak basis for a general text-teacher claim.**  
   `STEM-GNN/scripts/llm_stem/lowlabel_controls.py:75,77`  
   The teacher uses the same training labels and permitted validation labels. However, its checkpoint is selected on V21 accuracy and its temperature subsequently selected on those same 21 nodes. This is permitted validation reuse, with substantial selection noise and overfitting risk. It is also a MiniLM-based teacher, so “No LLM” should mean “no Qwen,” rather than no pretrained language knowledge.  
   **Fix:** narrow the comparator wording, save teacher best epoch and validation NLL, and preregister a simpler teacher sensitivity check if broader teacher-superiority claims matter.

8. **SHOULD-FIX — The selfcheck can reject a correct null result.**  
   `STEM-GNN/scripts/llm_stem/lowlabel_controls.py:162`  
   An active regularizer need not change any argmax predictions. Requiring every treatment to change predictions makes successful launch depend on observing an effect.  
   **Fix:** check finite, nonzero treatment loss/gradient or changed logits on a controlled fixture; report argmax changes descriptively.

9. **NIT — Selfcheck repeats substantial training and inference.**  
   `STEM-GNN/scripts/llm_stem/lowlabel_controls.py:165,166`  
   `predictions(models2, ...)` evaluates all seven arms seven times. Then `run_cell` retrains V21 after two complete V21 fits, alongside V500.  
   **Fix:** compute `p2` once and reuse existing V21 results for row/reproduction checks.

10. **SHOULD-FIX — Empty teacher subsets silently poison the diagnostic.**  
    `STEM-GNN/scripts/llm_stem/analyze_lowlabel_controls.py:91`  
    An empty correct or wrong subset produces NaN; `np.mean` then makes the whole reported aggregate NaN.  
    **Fix:** save numerator/denominator counts, report empty subsets explicitly, and report how many cells contribute to each conditional mean.

Checks that **passed static inspection**:

- V500 reproduces Stage 2’s intended S0/S-A recipe: identical low-label seed/mask, calibration inputs, anchor mask, finetune seed, and encoder reuse. MLP-M uses the same `base + 13` seed; QWEN argmax is unchanged by positive temperature.
- S0 correctly receives `prior=None, anchor_mask=None`; Stage 2 ignores the anchor when no prior exists.
- `fit_mix` handles V21 ties deterministically by choosing the smallest α. Coarse 1/21 accuracy increments make ties common, but this is documented behavior.
- The teacher scrambling check is valid **conditional on fixed masks, class count, and prior logits**: `calibrate` reads `y[va]`; MLP fitting reads training labels and validation accuracy. `y.max()` reads global class metadata, preserved by the balanced visible labels. The check does not certify the upstream audit or label-based mask construction.
- Test nodes in anchors are legitimate declared transductive distillation.
- Analysis pairing, percentage-point conversion, Holm implementation, drift threshold (`0.005`), and `_preds.pt` filename construction are correct.

**NO-GO.** Resolve checkpoint deserialization and preregister the narrower—or strengthened—claim rules before running.