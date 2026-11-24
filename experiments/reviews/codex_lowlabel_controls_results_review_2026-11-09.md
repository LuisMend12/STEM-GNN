The numerical results check out. I independently recomputed the arm means/SDs, reported differences, confidence intervals, wins, temperatures, mixture weights, and split means from all 15 JSONs. The Stage 2 reproduction is exact, the current training-code hash matches the JSONs, and the five-test Holm adjustment is correct: H90–H91 supported, H92–H94 inconclusive, Claim A supported, Claim B not met. Conditional accuracies match the raw output but cannot be independently recomputed without the cluster’s prediction files.

The blockers are interpretive, not numerical. All line references below are in `experiments/llm-stem/analysis_lowlabel_controls.md`.

1. **BLOCKER — Lines 21–26 and 87–90: prior quality is a plausible explanation, not an isolated cause.**

   “Not the validation size” is too strong: S-A loses 3.95 points under V21. S-A versus S-A0 changes the prior logits and potentially their selected temperature; prior accuracy does not describe all differences in target entropy, calibration, or errors. V500 does hold both Qwen temperatures at 2.0, strengthening the evidence for a prior-construction effect, but does not establish the proposed mechanism. The V21 comparison was prespecified as descriptive; the categorical headline promotes it beyond that status.

   **Replace lines 21–26 with:**

   > - **Descriptively, the rotation-averaged prior performs better than rotation 0 under both validation regimes.** Its standalone accuracy is 62.4% versus 53.5%. Under V21, S-A exceeds S0 by +5.66 [+2.25, +9.08], with 12/15 positive cells, but S-A itself loses 3.95 points relative to V500. These results are consistent with prior construction contributing to the difference; they do not establish prior quality as the sole explanation or rule out validation effects. S-A inherits the audit’s use of 30 labels, including some test nodes in other splits.

   **Replace lines 87–90 with:**

   > 2. **Prior construction is a candidate explanation.** S-A’s descriptive gain over S0 is +6.98 points in V500 and +5.66 in V21; S-A0’s corresponding differences are −1.35 and −0.97. These comparisons change the prior logits and, in V21, can also change the selected temperature. They do not isolate standalone teacher accuracy as the mechanism. The averaged prior was selected through a label-using audit and therefore does not establish the strict-budget claim.

2. **BLOCKER — Lines 72–76: conditional accuracy does not establish teacher following or explain S-A’s advantage.**

   Qwen-correct and MLP-correct observations are different subsets. Shared node difficulty can produce this pattern even without distillation: the output reports S0 at **85.4% / 42.2%** on MLP-correct/wrong observations, close to S-T’s **87.0% / 40.1%**. Accuracy when the teacher is wrong also does not measure whether student and teacher choose the same wrong class.

   **Replace lines 72–76 with:**

   > - **Student accuracy conditional on teacher correctness** (pooled node-cell observations): S-A scores 87.7% on Qwen-correct observations and 47.4% on Qwen-wrong observations in V500; S-T scores 87.0% and 40.1% on its own teacher’s corresponding subsets. These conditioning sets differ between teachers, so the rates cannot establish why S-A outperforms S-T. Shared node difficulty is a possible explanation: S0 already scores 85.4% / 42.2% on the MLP teacher’s subsets. Matched-subset comparisons against S0 and student–teacher agreement, including agreement on wrong labels, would help assess reliance but would not alone establish causality.

3. **SHOULD-FIX — Lines 14–17 and 84–86: narrow “specificity” to the tested recipes.**

   “Among the tested controls” makes a narrow empirical reading defensible. However, “real,” “isn’t a generic soft-target effect,” and “carries information … MiniLM don’t” imply more than these controls establish. Uniform differs in entropy; S-T is one untuned recipe; S-T versus S0 is not an equivalence test. Also, −0.05 is the **distilled student’s** difference, not the teacher-alone result.

   **Replace lines 14–17 with:**

   > - **With V500, S-A outperforms both tested control recipes.** It exceeds uniform-target distillation by +10.68 points and the specified MiniLM-MLP distillation recipe by +7.03, both in 15/15 cells and Holm-significant (H90, H91 supported). Descriptively, S-U and S-T differ from S0 by −3.70 and −0.05 points. This supports superiority over these controls, not a general claim about Qwen-specific information or other soft targets and text teachers.

   **Replace lines 84–86 with:**

   > 1. **The Stage 2 result survives these two specified controls under V500.** Uniform-target regularization and this MiniLM-MLP distillation recipe do not reproduce S-A’s observed improvement. Entropy-matched controls and stronger or differently tuned text teachers remain untested.

4. **SHOULD-FIX — Lines 91–93: define a class-count-independent rule and the new label budget.**

   Pre-fixing the method before an untouched second dataset is sound. Literal offsets `{0,2,4}` do not transfer unchanged across class counts: offsets wrap modulo the number of classes and may duplicate orderings. Likewise, three labels per class means `3C + 3C`, not necessarily `21 + 21`.

   **Replace lines 91–93 with:**

   > 3. **Next test:** before examining a second dataset’s outcomes, preregister a class-count-independent rule—for example, average class-aligned logits over every distinct cyclic option rotation, then select temperature using only the designated validation labels. Use three training and three validation labels per class (3C + 3C labels for C classes), and freeze the comparisons, thresholds, multiplicity correction, and sample design in advance. This would test a newly specified pipeline on another dataset, rather than literally replicate Cora’s {0, 2, 4} subset. Further evaluation on the already inspected Cora splits cannot provide independent confirmation.

5. **SHOULD-FIX — Lines 66–69: V21 changes anchor membership as well as validation size.**

   `train_regime()` uses `anchor = ~(low | va)`. Thus the other 479 validation nodes enter the KL objective under V21. The regime comparison combines selection/calibration effects with an enlarged distillation set.

   **Replace lines 66–69 with:**

   > - **Validation-regime differences (V21 − V500), descriptive:** S0 −2.63, S-A −3.95, S-A0 −2.26, S-U −2.28, S-T −5.82, MIX −5.69, and MLP-M −2.90 points. Every trained arm’s mean accuracy decreases. These comparisons combine reduced validation information with, for KL arms, adding the other 479 validation nodes to the unlabeled anchor set. QWEN and QWEN0 predictions are unchanged because temperature scaling preserves their argmax.

6. **SHOULD-FIX — Lines 97–103: add the transductive and label-acquisition qualifications.**

   The present limitations omit two important scope constraints. Test-node text and teacher predictions participate in training. Also, constructing exactly class-balanced subsets uses labels in the original pools; “42 labels” describes retained supervision under this benchmark sampling scheme, not demonstrated acquisition from 42 initially unlabeled queries.

   **Insert after line 103:**

   > - This is transductive evaluation: test-node text and teacher targets enter KL training. It does not establish performance on unseen nodes or graphs.
   > - The label budget counts retained training and validation labels under class-balanced benchmark sampling. Constructing these subsets uses class labels in the original pools; acquiring such balanced subsets from initially unlabeled nodes may require additional labeling.
   > - Only one Qwen model, one fixed KL weight, and one MiniLM-MLP teacher recipe were tested.

7. **SHOULD-FIX — Lines 8–10: distinguish local verification from reported cluster checks.**

   The repository supports direct verification of the JSON reproduction, but contains no prediction `.pt` files. The analysis code does assert prediction-level reproduction, and the raw output ends successfully; that is evidence of a reported cluster check, not an independently repeatable local check.

   **Replace lines 8–10 with:**

   > **Reproduction.** The V500 JSON accuracies for S0, S-A, MLP-M and QWEN exactly match Stage 2 in every cell (max drift 0.000 points). The cluster analysis completed successfully and includes assertions that saved predictions reproduce every reported accuracy. Prediction files remain on the cluster, so prediction-level checks and conditional-accuracy diagnostics cannot be independently recomputed from this repository alone.

8. **SHOULD-FIX — Lines 89–90: remove or substantiate the uncited “standard way” claim.**

   It is acceptable to describe the procedure’s intended purpose; calling it standard and implying that bias reduction occurred here requires supporting literature and a distinction between general precedent and this experiment. Standalone accuracy improvement is not itself a measurement of position bias.

   The replacement in finding 1 removes this issue. **If retaining a separate sentence, use:**

   > Rotation averaging combines predictions across option orderings; this experiment does not directly measure whether it reduces answer-position bias.

9. **NIT — Lines 25–26: clarify what the audit actually selected.**

   `stage2.audit()` measures the average across **all seven rotations**, then uses that gain to choose between the fixed three-rotation subset and rotation 0. It does not directly measure the three-rotation subset’s audit gain. “Fully label-free” is also misleading because temperature calibration still uses validation labels.

   **If retaining this passage independently, replace it with:**

   > The audit used 30 labeled nodes to choose between rotation 0 and the fixed {0, 2, 4} subset, using the measured gain from averaging all seven rotations as its decision statistic. An audit-label-free averaged-prior pipeline, with temperature selected only on the designated validation set, has not yet been tested independently.

10. **NIT — Line 32: identify the intervals as unadjusted.**

   The p-values receive Holm adjustment; the displayed intervals are ordinary paired-t intervals, not simultaneous familywise intervals.

   **Replace the table header with:**

   > | ID | Comparison | Regime | Mean [unadjusted 95% CI] | Wins | Holm p | Verdict |

An accurate plain-language summary would be:

On Cora with 21 training labels and 500 validation labels, the rotation-averaged Qwen-prior student outperforms uniform-target distillation and the specified MiniLM-MLP distillation recipe under the prespecified tests. The strict 21-training-plus-21-validation-label claim is not established: the rotation-0 Qwen arm has inconclusive comparisons against all three baselines. The averaged-prior arm retains a descriptive advantage with small validation, but inherits the earlier label-using audit. These results suggest that prior construction matters, without establishing teacher accuracy as the cause or ruling out validation effects. An independently evaluated second dataset, using a preregistered rotation rule and label budget appropriate to its class count, is a reasonable next test.