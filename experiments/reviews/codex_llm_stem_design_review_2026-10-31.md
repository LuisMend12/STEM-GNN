**The protocol is not ready to launch.** S0 is an incompletely specified, modified STEM baseline; S-R needs an API change; and the existing pretraining and attack wrappers cannot implement the proposed MoE study unchanged. S-V is plausible only with an explicit gradient path and a narrower claim.

I only read files. No files were changed and no experiments were run.

Below are concrete amendments, ranked by severity. Citations are relative to the workspace.

1. **BLOCKER — Define S0 completely and distinguish architectural fidelity from recipe fidelity.**

   The proposed MoE forward is real: experts concatenate the neighbor mean and root features, apply expert matrices, optionally add a residual, then mix expert outputs with router probabilities. BatchNorm, activation and dropout follow the mixture. This is materially different from the plain SAGE path, including message ReLU, bias and residual behavior. See [encoder.py:154](STEM-GNN/model/encoder.py:154) and [encoder.py:420](STEM-GNN/model/encoder.py:420).

   However, the protocol never fixes **depth, hidden width, normalization, dropout, routing temperature, optimization, pretraining losses or decoder mode**. These cannot be inherited implicitly:

   - The root pretraining configuration specifies depth 2, width 768, commitment weight 10 and additional reconstruction objectives: [pretrain.yaml:3](config/pretrain.yaml:3).
   - X2 uses depth 2, commitment weight 0.25, and feature-plus-topology reconstruction: [run.py:167](STEM-GNN/scripts/stem_fixes/run.py:167).
   - The earlier anchor explicitly describes itself as a component pilot, not the complete architecture: [protocol.md:73](experiments/llm-anchor/protocol.md:73).

   **Amendment:** publish one complete configuration and call S0 “STEM components with EMA repair and the declared pretraining recipe.” If claiming the full STEM objective, retain and specify its semantic and topology-semantic losses, rather than silently substituting X2; those objectives exist in [pt_model.py:116](STEM-GNN/model/pt_model.py:116).

   **Depth recommendation:** use depth 2, width 128 for the main mechanism study; reserve depth 16 for a secondary stress test. Width-64 MoE fell from 80.0 to 74.3 with depth, while width-128 stacked initialization remained viable at depth 16. Per-expert initialization was disastrous under the tested depth-16 recipe, but this is not evidence that every width or optimizer behaves similarly. See [moe-depth analysis:13](experiments/moe-depth/analysis.md:13), [stem-fixes analysis:18](experiments/stem-fixes/analysis.md:18), and [stem-fixes analysis:50](experiments/stem-fixes/analysis.md:50). **Deep MoE is unlikely to be a clean starting point for isolating an LLM mechanism.**

2. **BLOCKER — S-R cannot use a separate Qwen input through the current router API.**

   S-R0 works as described: `use_llm_router=True` builds every router with `input_dim` inputs and routes on `x_orig=x`. Thus MiniLM can remain both the expert input and the structure-free router input. See [encoder.py:218](STEM-GNN/model/encoder.py:218) and [encoder.py:412](STEM-GNN/model/encoder.py:412).

   S-R does **not** work unchanged:

   - `x_orig` is always the expert input, not an independently supplied tensor.
   - With MiniLM expert inputs, the routers expect the MiniLM dimension, not 1,536.
   - `aux_router_feat` is concatenated only at layer zero when `use_llm_router=False`; it does not replace router inputs at every layer.
   - The main finetuning constructor does not pass `use_llm_router` at all: [finetune.py:124](STEM-GNN/finetune.py:124).

   **Amendment:** introduce independent `router_input_dim` and `router_features` arguments, propagate them through pretraining, finetuning, evaluation and attack forwards, and assert their shapes. Decide explicitly whether routing sees clean text features while expert features are masked during pretraining.

   Also distinguish **fixed router inputs** from **frozen router parameters**. S-R proposes edge-invariant inputs; its router weights can still learn.

3. **BLOCKER — Build MoE pretraining rather than reusing `pretrain_x2` unchanged.**

   `pretrain_x2` explicitly constructs a two-layer, non-MoE encoder. It cannot produce pretrained expert and router weights for S0/S-R: [run.py:179](STEM-GNN/scripts/stem_fixes/run.py:179). The anchor calls this function directly: [run.py:197](STEM-GNN/scripts/llm_anchor/run.py:197).

   The X2 feature/topology objective is mathematically usable with MoE; the implementation must instantiate that MoE and forward separate router features. Specify whether the environment regularizer is included: the full pretraining path includes it, while X2 does not include it in its loss. See [pretrain.py:56](STEM-GNN/pretrain.py:56) and [run.py:201](STEM-GNN/scripts/stem_fixes/run.py:201).

   **Amendment:** require identical encoder architecture and router semantics between pretraining and finetuning, strict checkpoint loading, and explicit loss coefficients. Do not initialize a new MoE around a SAGE checkpoint and call that equivalent pretraining.

4. **BLOCKER — Verify the entire attacked model, not only the aggregation.**

   The proposed direction is correct: MoE aggregates `x[col]` into `row`, whereas the existing SAGE wrapper aggregates `relu(x[src])` into `dst`. See [encoder.py:167](STEM-GNN/model/encoder.py:167) and [attack run.py:54](STEM-GNN/scripts/code_robustness/run.py:54).

   A faithful weighted MoE relaxation is:

   \[
   a_i=\frac{\sum_{e:\,\mathrm{row}(e)=i}w_e x_{\mathrm{col}(e)}}{\sum_{e:\,\mathrm{row}(e)=i}w_e},
   \]

   with a defined zero-degree output, followed by the original concatenation, expert transforms, residual, routing and normalization. **Multiplying messages by weights and then taking an ordinary scatter mean is wrong**: zero-weight edges would still count in the denominator.

   **Amendments:**

   - Test all-one, binary deletion, additions and isolated-node cases.
   - Include asymmetric toy graphs: symmetric Cora graphs can conceal a direction error.
   - Match intermediate expert outputs, routing, representations, code assignments and final logits against the actual evaluation model.
   - Respect `normalize='none'`: the current attack wrapper unconditionally applies BatchNorm, unlike the real encoder’s conditional normalization. See [attack run.py:62](STEM-GNN/scripts/code_robustness/run.py:62).
   - Match the chosen decoder path. `TaskModel` can decode concatenated per-head codes or projected quantized output; the attack wrapper implements only the former. See [ft_model.py:140](STEM-GNN/model/ft_model.py:140).

   The proposed “nonzero gradient through router weights” assertion is insufficient. Parameter gradients do not establish an edge-to-router gradient path. Check derivatives of **later-layer router logits with respect to edge weights**, and compare differentiable intermediate quantities with finite differences. The first-layer S0 router is already edge-invariant.

5. **HIGH — Own-model PRBCD is necessary, but does not establish attack adequacy.**

   Attacking S-R directly is correct. Holding Qwen router features constant is also correct for an edge-only threat model; the attack must still differentiate through every expert aggregation and subsequent representation.

   Softmax routing is differentiable and does not inherently imply masking, although saturation can weaken gradients. VQ is the larger concern: evaluation uses discrete lookup, while the harness substitutes identity STE or expected-code backward gradients. Its soft backward temperature is fixed at 0.1. See [vq.py:655](STEM-GNN/model/vq.py:655) and [attack run.py:95](STEM-GNN/scripts/code_robustness/run.py:95).

   **Amendment:** retain hard-forward evaluation, but add prespecified BPDA temperature variants, more restarts, margin-based objectives, stronger-search convergence checks, MoE continuous-surrogate transfer and discrete local refinement. Check whether stronger search consistently lowers accuracy. Use EOT only if inference actually remains stochastic; ordinary evaluation here is deterministic.

   Freeze codebooks, BatchNorm statistics and all model state throughout attack generation. The shared candidate union is valuable, but **a union of weak attacks remains weak**.

   Declare the attack’s label access: the current harness passes test labels and test indices to PRBCD. This is a label-informed evaluation attacker, not training-label leakage: [run.py:210](STEM-GNN/scripts/code_robustness/run.py:210).

6. **HIGH — Resolve the edge constraint explicitly for text-conditioned routing.**

   No listed arm explicitly constructs an LLM-dependent adjacency, selects neighbors or changes graph degree. S-F, S-V and S-A operate on representations or supervision.

   **S-R is the boundary case.** At a given node its effective neighbor transform is
   \[
   \sum_k p_k(q_i)W_{k,\mathrm{neighbor}},
   \]
   so text changes the matrix applied to every incoming neighbor contribution. It does not assign different scalar weights to individual edges, but it does modulate messages.

   **Amendment:** define the constraint operationally: adjacency, edge existence, scalar aggregation coefficients and sampling probabilities must be independent of text; node-conditioned expert transforms are allowed. Under that interpretation, the proposed arms comply. Under a literal prohibition on any text-dependent modulation of edge contributions, **S-R and its combined arms violate the constraint**. The protocol must not silently assume these meanings are interchangeable.

   Topology reconstruction may predict edge scores as an auxiliary loss, but those scores must never feed back into adjacency or message weighting.

7. **HIGH — Rewrite S-V around its actual gradient mechanism.**

   “Assigned codes” is ambiguous: integer indices, detached lookup vectors, the straight-through concatenated code tensor, or projected quantized output have different gradient paths.

   The known defect matters:

   - Reconstruction gradients pass through the training STE to encoder-side representations, rather than directly updating selected code vectors.
   - Commitment targets are detached under the default non-learnable codebook setting.
   - EMA updates code vectors from assigned encoder representations.

   See [vq.py:929](STEM-GNN/model/vq.py:929), [vq.py:986](STEM-GNN/model/vq.py:986), and [vq.py:661](STEM-GNN/model/vq.py:661).

   Therefore, **a detached hard-code predictor is unlikely to shape the encoder or codebook at all**; it may only train the auxiliary predictor. With the returned training STE tensor, semantic supervision can shape the encoder and indirectly move EMA centroids. It still does not directly optimize code vectors by reconstruction gradient.

   **Better placement:** define S-V as semantic distillation at the quantizer input: predict a normalized Qwen target from `vq.project_in(z)`, retain commitment and EMA, and discard the predictor afterward. This gives an unambiguous differentiable mechanism. Compare against a continuous/no-VQ model with the same semantic loss to distinguish representation distillation from a VQ-specific benefit. If using post-quantization STE instead, state the exact returned tensor and gradient path.

   X2 itself found the no-VQ reference above C-ema, so retaining VQ needs justification: [analysis.md:63](experiments/stem-fixes/analysis.md:63).

8. **HIGH — Correct the coefficient provenance and tuning rule.**

   A root config directory **does exist**. Its Cora finetuning section contains learning-rate and stopping settings, but no Jacobian/Lipschitz coefficients: [finetune.yaml:1](config/finetune.yaml:1). Both coefficient arguments default to **0.0**, as do `TaskModel`’s parameter lookups: [args.py:153](STEM-GNN/utils/args.py:153), [ft_model.py:37](STEM-GNN/model/ft_model.py:37). I found no configured nonzero Cora values in the searched model/config paths.

   The name matching **does catch** `MixtureSageLayer.weights`: `"weight"` is a substring of `"weights"`, and the tensor is three-dimensional. It also catches MySAGEConv’s `lin_l.weight` and `lin_r.weight`. See [encoder.py:161](STEM-GNN/model/encoder.py:161), [encoder.py:101](STEM-GNN/model/encoder.py:101), and [encoder.py:474](STEM-GNN/model/encoder.py:474).

   However, it excludes routers, normalization and VQ projections. These are Frobenius penalties, **not an end-to-end structural robustness guarantee**.

   **Amendment:** call nonzero values study-selected hyperparameters. Prespecify a small grid, selection metric, split policy and tie rule; select using clean validation only and freeze before arm comparisons. Include zero. If zero wins, the protocol cannot simultaneously claim both penalties are active. Shared coefficients are reasonable for a controlled intervention study, but are not equally tuned optima—and raw summed penalties change scale with width and parameter count.

   Finally, the anchor training loop only adds CE and KL; reusing it would omit both penalties entirely: [run.py:91](STEM-GNN/scripts/llm_anchor/run.py:91).

9. **HIGH — Make the primary comparison match the proposed mechanism and reduce scope.**

   S-ALL versus S0 measures a bundle, not the router explanation. It can improve through input-independent distillation while routing contributes nothing. The one-at-a-time arms estimate effects **at S0**, not contributions inside S-ALL. See [protocol.md:25](experiments/llm-stem/protocol.md:25).

   **Amendment:** make S-R versus S-R0 the primary comparison for a Qwen-specific routing claim, with S-R versus S0 as a prespecified mechanism comparison. Make S-ALL exploratory. If the question is instead practical full-stack performance, H50 is acceptable, but abandon causal placement attribution.

   For interactions, either run the R×V×A factorial or add leave-one-out combined arms. Do not infer synergy from S-ALL exceeding individual arms. S-ALL+F versus S-ALL estimates the input effect only in that combined context.

   The zero routing-shift diagnostic is an implementation check, not evidence that routing causes robustness. Lower expert-output shift is not logically necessary for accuracy improvement.

10. **MEDIUM — Tighten inference, leakage controls and compute commitments.**

   **Statistics.** Fifteen split/seed cells are not fifteen independent graphs or fifteen independent label splits. Keep the CI explicitly conditional and report all three split-level effects. The earlier protocol already required this: [llm-anchor protocol:124](experiments/llm-anchor/protocol.md:124).

   Prespecify Holm’s exact family, directional tests and treatment of H53’s two endpoints. Distinguish “point estimate exceeds +2 and CI excludes zero” from evidence that the true effect exceeds +2. For the clean-loss guardrail, use a declared non-inferiority criterion if claiming preserved clean accuracy. Retain an inconclusive region.

   Define worst-node as correctness intersected across the finite candidate union; it can combine failures from different graphs and is not accuracy on one realized attack.

   **Compute.** There are eight graph arms including S0. Across 15 cells and two label settings, that is **240 graph-model finetunes** before controls. Three own-model attacks at two budgets give **1,440 PRBCD calls**; MIX adds another 180. Each current call uses 100 optimization epochs and block size 100,000: [run.py:216](STEM-GNN/scripts/code_robustness/run.py:216). Union evaluation grows approximately quadratically with the number of attacked models. CPU affordability is unestablished, especially at depth 16 or width 768. Stage the router study first and set a wall-time budget before authorizing the full matrix.

11. **MEDIUM — Keep pooling, but strengthen feature and calibration provenance.**

   `qwen_features.mean_pool` correctly masks padding and divides by the number of real tokens. It restores outputs by node ID and extracts without labels: [qwen_features.py:21](STEM-GNN/scripts/llm_robustness/qwen_features.py:21). This is valid pooling, though not proof that mean causal hidden states are strong sentence embeddings. Add empty-input handling and pin model/tokenizer revisions.

   The dimension is handled by X2’s `x.size(1)` encoder and reconstruction decoder, and the MLP also accepts a dynamic input dimension. The general entry points instead use configured `input_dim`; the separate Qwen router path remains unsupported as described above. See [run.py:182](STEM-GNN/scripts/stem_fixes/run.py:182) and [finetune.py:124](STEM-GNN/finetune.py:124).

   Important leakage qualifications:

   - The prior prompt uses category names, not node labels; labels are used afterward to print accuracy over all scored nodes: [qwen_prior.py:88](STEM-GNN/scripts/llm_robustness/qwen_prior.py:88), [qwen_prior.py:131](STEM-GNN/scripts/llm_robustness/qwen_prior.py:131). Thus the reported 53.5% has already exposed aggregate test performance. Record that exposure and freeze the design now.
   - Validation-calibrated priors are then used on validation nodes in the KL loss. This is validation-informed supervision, not wholly label-free supervision. Prefer separate calibration and checkpoint-selection subsets, or disclose their reuse.
   - Keep test labels out of calibration, rotation selection and arm selection.
   - Validate cache text hashes and node order, not only dimensions. The existing loader checks version, permutation and shape but not text hash: [run.py:52](STEM-GNN/scripts/llm_anchor/run.py:52).
   - Preserve the description “21 training labels with the original validation budget” for low-label results: [protocol.md:118](experiments/llm-anchor/protocol.md:118).

The following controls separate the main alternative explanations:

| Apparent benefit | Necessary separating control |
|---|---|
| Qwen is simply a better feature extractor | MiniLM/Qwen feature swap crossed with MLP and plain SAGE; matched projection dimension and tuning budget |
| Removing edge-sensitive routing helps regardless of Qwen | S-R0, uniform routing, and learned global routing |
| Random node partitions regularize experts | Fixed random-feature router and node-permuted Qwen-feature router |
| Baseline router is weak or poorly optimized | Softmax-training router and matched router capacity/optimization |
| Semantic auxiliary supervision helps, rather than VQ | Same auxiliary loss without VQ, plus a MiniLM-target control |
| Head anchoring merely copies the teacher | Directly attacked MIX, MLP+anchor, MiniLM teacher, and Qwen-wrong-node analysis |
| Quantization hides usable attack gradients | Continuous counterpart, multiple BPDA settings, transfer and discrete refinement |
| Larger inputs change capacity or regularization | Matched bottleneck width and reported parameter counts |

**My recommendation:** start with a fully specified shallow MoE study of S0, S-R0 and S-R, plus uniform/random-routing controls. The separate router mechanism is implementable and plausible, but its Qwen-specific advantage is unproven. The current S-V wording and the full-stack primary would make positive results much harder to interpret.
