The review identifies real problems, but **F1’s accuracy severity is overstated, F3 contains a factual error, and F4 misidentifies the regularizer**. The largest omissions concern VQ learning, directed KG readout, and pretraining augmentation.

Everything below is based on read-only inspection and short JSON/raw-triple calculations. No files were modified and no training was run. Code citations are relative to `STEM-GNN/`.

**(a) Verification of F1–F7**

| Finding | Correct / overstated? | Evidence file:line | Corrected text to apply |
|---|---|---|---|
| **F1: edge-feature handling** | **Mechanism correct; HIGH accuracy severity unproven.** The width claim has exceptions. “Whenever `--moe` is on” is inaccurate. | `model/encoder.py:127`, `:139–142`, `:167–174`, `:446–450`; `utils/args.py:31–32` | “SAGE adds unprojected edge embeddings before message ReLU at every layer. With multiple layers and edge attributes, deeper additions require compatible hidden/edge widths. A one-layer encoder can have a different output width. Active MoE layers ignore edge attributes; hybrid encoders can still consume them through SAGE layers. `--moe` alone does not activate experts when `moe_layers='none'`. Per-layer edge projections are a sound compatibility/information-access fix, but E1 provides no demonstrated Cora accuracy improvement.” |
| **F2: aggregation direction** | **Correct. MEDIUM is reasonable for directed data.** | `model/encoder.py:86`, `:127`, `:168–169`; `dataset/data/KG/gen_data.py:71` | “MoE reads outgoing neighbors, whereas default-flow SAGE reads incoming neighbors. These agree on reciprocal graphs. Standardize direction before attributing differences to expert routing.” The reported WN18RR train reciprocity is correct: **34.38% of triple rows** have a reversed endpoint pair, ignoring relation identity. |
| **F3: unmatched MoE operator** | **Mostly correct, but first-layer residual statement is wrong for default widths.** Initialization deserves more emphasis. | `model/encoder.py:160–173`, `:211–227`, `:447–456`; `utils/args.py:13–14` | “Residuals are present whenever input and output dimensions match, including the first layer at the default 768→768 width. They were absent in the earlier 384→64 experiment. MoE also differs in message ReLU, bias, initialization, and residual placement. Match these before interpreting MoE/SAGE accuracy differences.” Stacked Xavier initialization is especially consequential; see below. |
| **F4: entropy versus balancing** | **Correct that it is not batch load balancing; incorrect to call it exactly negative entropy.** Default-off severity should be LOW/conditional. | `model/encoder.py:256–258`, `:430–437`; `utils/args.py:35` | “The loss is `mean(sum(w_gumbel * log_softmax(logits)))`, with gradients through both factors. It becomes negative entropy if `w=softmax(logits)`, but training uses Gumbel-softmax samples. It is a stochastic per-node routing regularizer, not a batch-usage balance objective. A batch-importance loss is a reasonable alternative; any sharpness term needs separate justification.” |
| **F5: training-mode EMA teacher** | **Correct behavior; accuracy harm remains a hypothesis. Proposed fix is incomplete.** | `pretrain.py:26`, `:66`; `model/pt_model.py:22`, `:93`, `:104–106`; `model/encoder.py:431–437` | “The detached teacher still uses dropout, Gumbel routing, and training-mode BatchNorm. EMA updates parameters but not buffers. Test deterministic teacher targets, while explicitly copying or averaging BatchNorm buffers—or using a normalization without running statistics. Merely calling `eval()` would otherwise leave teacher statistics stale.” |
| **F6a: discarded VQ loss** | **Correct, and broader than node finetuning. MEDIUM is plausible, not established empirically.** | `model/ft_model.py:132–146`; `task/node.py:74`; `task/link.py:36`; `task/graph.py:36` | “All task paths discard the VQ auxiliary loss. That returned loss includes commitment and potentially orthogonal regularization, so expose commitment separately for a clean ablation. Frozen VQ parameters do not eliminate commitment gradients to the encoder. The proxy attack gain is exploratory and does not establish a full-model benefit.” |
| **F6b: eval gradients** | **Correct mechanism, but ‘sees zero gradients’ is too absolute.** | `model/vq.py:655–659`, `:929–937`; `model/ft_model.py:142–146` | “Eval-mode hard assignment disconnects the input-to-code gradient. Depending on the attack implementation, this yields zero gradients or an unused/disconnected-gradient error. Use an attack-only BPDA/STE wrapper while keeping BatchNorm and routing in eval mode. The approximately 12-point gap belongs to the proxy experiment.” |
| **F7: reconstruction width** | **Correct. LOW for default settings; a real configuration failure otherwise.** | `pretrain.py:123`; `model/pt_model.py:80–81` | “The edge-semantic decoder must output the edge-embedding dimension. That equals `input_dim` in the current data pipeline, but should ideally be represented explicitly as `edge_dim`.” |

Additional qualifications:

- Cora’s literal edge-text string is at **`dataset/data/single_graph/Cora/gen_data.py:57`**, rather than line 56.
- Asking an encoder to reconstruct attributes it did not receive is not intrinsically contradictory: it can infer them from endpoints/topology. F1b’s stronger argument is the lost ability to condition messages on informative bonds.
- Joint neighbor/root matrices are not inherently less expressive than separate matrices. With uniform routing, MoE reduces to an effective linear neighbor/root operator, with its residual absorbed into the root matrix.
- A constant edge term is a bias **inside a nonlinear message function**. Its effect cannot generally be removed exactly by a later affine map or BatchNorm.

**(b) Important missed issues, ranked by likely accuracy/generalization impact**

Ranks reflect expected impact and scope, not demonstrated accuracy gains.

| Rank | Missed issue and likely impact | Evidence file:line | Falsifiable test |
|---|---|---|---|
| **1 — High, across pretraining** | **Default codebooks do not learn to fit the data.** Both constructors disable EMA and leave outer `learnable_codebook=False`. Orthogonal regularization makes the inner embeddings parameters, but commitment detaches them and reconstruction uses a detached STE difference. Their loss-derived gradient therefore comes from orthogonal regularization, rather than reconstruction/commitment. Weight decay can also move them. With orthogonal weight zero, they become fixed random buffers. | `pretrain.py:105–118`; `model/vq.py:722`, `:747`, `:785`, `:931–937`, `:1007–1028` | Backpropagate each pretraining loss separately and inspect codebook gradients. Prediction: reconstruction and commitment give no embedding gradient; orthogonal loss does. Then compare a coherent EMA codebook against a fixed dictionary with orthogonal regularization controlled. |
| **2 — High on FB15K237** | **Endpoint averaging destroys relation direction before classification.** Predictions for `(h,t)` and `(t,h)` are identical on a fixed encoded graph, regardless of encoder quality. It also loses other pair distinctions: different endpoint pairs can share a mean. | `task/link.py:7–8`, `:31`, `:68`, `:111` | Replace the mean with an ordered pair representation. Check reversed-pair predictions and compare matched CPU probes. Raw-data collision calculations below show substantial FB15K237 exposure. |
| **3 — High for directed KG pretraining** | **Augmentation forces directed edges into an undirected view.** `dropout_adj(...force_undirected=True)` selects edges by endpoint ordering and duplicates retained edges in reverse. This can discard one-way KG edges according to arbitrary entity IDs and invent reverse edges, while reconstruction targets remain directed. | `pretrain.py:42–44`; `dataset/data/KG/gen_data.py:71`; `model/pt_model.py:128–131` | Apply augmentation to a one-way edge in each orientation and at `edge_p=0`. Check retention and invented reverses. Compare directed dropout against current augmentation with unchanged losses. |
| **4 — Medium–High, active MoE** | **Stacked Xavier initialization strongly suppresses expert transforms.** Applying Xavier to `[K,2d,d]` uses tensor fan dimensions, not independent matrix initialization. For equal widths, variance is approximately `2/[d(2d+K)]`; independently initialized `[2d,d]` matrices get `2/(3d)`. At `d=768,K=3`, the former variance is about **1/513** as large. Equal-width layers consequently start close to their residual path. | `model/encoder.py:161–173` | Compare initialization-only transform/residual RMS and gradients. Then initialize each expert separately, controlling effective output scale and routing. |
| **5 — Medium, generalization across graphs/batches** | **Normalization choice is misleading, and final BatchNorm affects VQ assignments.** Every norm is `BatchNorm1d`; `'layer'` selects BatchNorm too. The final output is normalized before VQ. Sampled neighborhoods, whole graphs, and different dataset mixtures can produce different running-statistic requirements. | `utils/args.py:18`; `model/encoder.py:227`, `:452–456`; `utils/loader.py:10–26` | Verify module types under `'layer'`. Compare fixed seed-node representations under full and sampled neighborhoods. Implement actual LayerNorm and test transfer without changing other operations. |
| **6 — Medium, VQ semantics** | **The “cosine” codebook does not normalize learned embeddings at lookup.** Queries are normalized, but lookup uses `self.embed` directly. Initialization normalizes embeddings; non-EMA gradient updates need not preserve their norms. Orthogonal regularization normalizes internally and therefore does not enforce unit norms. | `model/vq.py:232–237`, `:536`, `:541–542`, `:648–650`, `:891` | Measure embedding norms and compare assignments using raw versus normalized embeddings. Prediction: unequal norms can change winners. Normalize at lookup and test assignment/accuracy effects. |
| **7 — Medium, finetune bottleneck** | **The four-head decoder is an overparameterized single linear classifier, not an independent-head ensemble.** Each output head reads all concatenated codes; averaging its logits is algebraically one linear map on those codes. Pretraining decodes `project_out(codes)`, while default finetuning bypasses that projection. | `model/ft_model.py:43–46`, `:133`, `:142–146`; `model/vq.py:1040–1041` | Average decoder row blocks into one matrix and verify identical logits. Compare matched classifiers on concatenated codes versus projected quantization. Do not interpret the current four outputs as independently trained voters. |
| **8 — Medium, pretraining objective quality** | **Losses include weak or contradictory supervision.** Constant edge targets admit a decoder-bias solution. Topology negatives are drawn after subsampling positives, so omitted true edges can become negatives. Feature reconstruction covers all seed features, including unmasked ones. Loss weights differ greatly, without establishing comparable gradient scales. | `model/pt_model.py:42–43`, `:51–60`, `:80–81`; `pretrain.py:41`, `:51–58`; `utils/args.py:46–50` | Count sampled negatives overlapping the original graph; measure per-loss encoder gradient norms; compare masked-only reconstruction and removal of constant-target edge reconstruction. |
| **9 — Conditional High for evaluation claims** | **KG evaluation includes validation/test edge existence in message passing.** This is transductive relation classification, not held-out-link prediction. Constant edge text avoids directly revealing relation labels, but held-out topology is available. Restoring relation-specific text naively would expose the target relation through `xe`. | `dataset/data/KG/gen_data.py:76–81`, `:90–93`; `utils/preprocess.py:10–13`; `dataset/process_datasets.py:104–107`; `task/link.py:20–29`; `utils/loader.py:29–47` | Compare all-edge versus train-edge message-passing graphs. If introducing relation attributes, verify that every query edge’s target relation is absent from its input graph. |
| **10 — Medium, optimization** | **Pretraining scheduler time units disagree.** The cosine horizon is configured in epochs but advanced every minibatch, producing repeated cosine cycles when the number of steps exceeds that horizon. | `pretrain.py:64–65`, `:136`; `utils/others.py:138–141` | Calculate the LR sequence from batch count without training. Use an epoch step or a total-step horizon and compare. |
| **11 — Medium for downstream evaluation** | **Finetuning reports best-validation metrics but does not restore those model weights.** Post-split callbacks receive the final model, which can differ from the reported best model. Logger ties also use the last equal-validation epoch, while stopping uses strict improvement. | `utils/early_stop.py:11–19`; `utils/logger.py:45`; `finetune.py:253–255`, `:279–283` | Compare callback predictions with predictions saved at the reported best epoch. Save and restore the selected state before callback evaluation. |

The VQ configuration also needs explicit documentation:

- With `heads=4`, `codebook_dim=code_dim=768`, the projected code space is **3072 dimensions**, with four separate 768-dimensional codebooks. It does **not** split 768 dimensions into four 192-dimensional heads (`model/vq.py:734–739`, `:776`).
- Finetune `kmeans_init=True` **does not overwrite a successfully loaded pretrained dictionary**. `load_params` performs a preliminary forward, then loads the full state—including `initted` and embeddings (`utils/others.py:167–170`; `model/vq.py:562–595`). Without a checkpoint, first-use initialization is data-dependent, even if parameters are frozen.
- The shared-decoder branch assumes `code_dim == hidden_dim`, although `quantize` has `hidden_dim` dimensions (`model/ft_model.py:46`, `:146`). Default settings conceal this configuration bug.
- The encoder loop correctly advances `z` through successive layers. Its ordering is **convolution/mix → BatchNorm → activation → dropout**, with no final activation/dropout. That is not itself a bug. Without edge attributes and with ReLU activation, later SAGE message ReLUs are largely redundant because preceding activation/dropout outputs are nonnegative.

**(c) E1 numerical, verdict, and implementation audit**

All **120 rows are unique and complete**: three splits × five seeds × two depths × four variants. Recomputed values match the analysis after rounding. SDs are sample SDs.

| Variant | Depth 2: accuracy %, mean ± SD | Depth 4: accuracy %, mean ± SD |
|---|---:|---:|
| S0 | 78.517 ± 1.580 | 78.330 ± 2.433 |
| S1 | 77.730 ± 1.742 | 78.562 ± 2.314 |
| S2 | 77.892 ± 1.550 | 78.765 ± 2.127 |
| S3 | 77.456 ± 3.031 | 77.176 ± 3.285 |

| Paired difference, percentage points | Depth 2: mean (SE) | Depth 4: mean (SE) |
|---|---:|---:|
| S1 − S0 | −0.7866 (0.5083) | +0.2321 (0.5490) |
| S2 − S1 | +0.1612 (0.3213) | +0.2031 (0.1793) |
| S3 − S1 | −0.2740 (0.7619) | −1.3862 (0.8825) |
| S3 − S0 | −1.0606 (0.5873) | −1.1541 (1.0879) |

H26 is **+0.0419 points, SE 0.3932**.

- **H24:** “No effect under the point-estimate rule” is correct.
- **H25:** “Refuted under the point-estimate rule” is correct because both differences are nonpositive.
- **H26:** “Not supported” is correct. The protocol uses strict `> +1`; the analysis renders `≥ +1`. This discrepancy does not affect the verdict.

These verdicts are threshold classifications, not equivalence tests or statistically resolved refutations. For example, a conventional paired-*t* interval for depth-2 S1−S0 is approximately **−1.88 to +0.30 points**, still permitting harm exceeding one point. Moreover, the 15 runs share one graph and only three splits; the reported SE is a run-level descriptive calculation, not evidence from 15 independent datasets.

The wrapper is mechanically sound on the inspected dense-`edge_index` path:

| Check | Assessment |
|---|---|
| Instance `layer.forward` override | Valid. Each closure captures its own bound original method and layer index. Module calls invoke the replacement; no extra `self` argument is needed. `scripts/architecture_review/edge_injection.py:66–76`. |
| S1 | Passes unchanged attributes through the real encoder and convolution. Correct reproduction of this encoder’s message behavior, though not the complete pretrained/VQ pipeline. |
| S2 | Passes `None` only after layer zero. Correct. |
| S3 | Registers trainable per-layer `Linear(edge_dim, layer.in_channels)` projections, including bias. Correct for the protocol’s formulation. |
| PyG `xe` propagation | `MySAGEConv` explicitly calls `_orig_propagate` when available and passes `xe=edge_attr`; `message` accepts and adds it. No inspected wrapper path drops `xe`. `model/encoder.py:124–142`. |
| Model selection | Validation selects and restores the state; test accuracy is computed afterward. Correct. `edge_injection.py:88–104`, `:129–131`. |

Important limitations and corrections to the interpretation:

1. **S3 has different stochastic trajectories.** Its extra linear-layer initialization consumes RNG before dropout. Shared seeds align initial encoder/head parameters, but not subsequent dropout draws (`edge_injection.py:61–65`, `:126–128`). Pairing remains useful, but is not identical-noise matching.
2. **S3 adds approximately 295,680 parameters at depth 2 and 591,360 at depth 4.** With constant edges, these parameterize per-layer message biases redundantly. A directly learned bias would be a cleaner, cheaper constant-feature control.
3. **The experiment changes the feature pipeline as well as width.** It uses raw `d.x` and MiniLM edge embeddings; the repo regenerates prefixed node-text embeddings using its configured sentence encoder (`edge_injection.py:35–39`, `:53`; `dataset/data/single_graph/Cora/gen_data.py:42`; `dataset/data_utils.py:26`). “This is what the repo feeds” should refer to the edge-text content and injection rule, not identical numerical inputs.
4. The JSON does not record feature norms, cache provenance, package versions, or code revision. The stated unit norms are not independently established by those result files.
5. **“BatchNorm and learned weights absorb the shift” is an untested mechanism.** The shift occurs before message ReLU, so exact cancellation is not assured.
6. Replace the headline interpretation with: **“This Cora proxy did not cross the preregistered harm threshold. Learned edge projection did not improve the point estimates. Neither result establishes equivalence or generalizes to informative edge attributes.”**

**(d) Single most promising next architectural test**

**Use an ordered endpoint representation for KG relation classification**, starting with FB15K237.

This has a directly demonstrable representational limitation and can be tested cheaply without molecules or full GNN pretraining. Counting all local raw triples, the maximum accuracy of an arbitrary classifier keyed only by unordered endpoint identity is:

| Dataset | Unordered-pair oracle ceiling | Ordered-pair oracle ceiling |
|---|---:|---:|
| WN18RR | 99.802% | 99.867% |
| FB15K237 | **84.810%** | **91.536%** |

These are **in-sample identity-collision ceilings**, not achievable test accuracies. Nevertheless, the current mean readout is necessarily bounded by unordered-pair identity on a fixed graph. FB15K237 exposes a **6.73-point structural ceiling gap**; WN18RR exposes little.

**Falsifiable prediction:** an ordered endpoint probe improves FB15K237 test relation accuracy by at least **2 percentage points** over the mean-endpoint probe, under matched validation selection.

**Cheapest decisive experiment:** freeze one cached/local text embedding per entity and compare:

- A: linear classifier on `(x_h + x_t)/2`;
- B: linear classifier on `[x_h, x_t]`.

Use official train/validation/test triples, identical seed sets and optimization budgets, and the same small validation-selected regularization grid. Include no relation text in inputs. This isolates endpoint ordering with CPU linear probes. A clear win warrants the same replacement in the real GNN/VQ pipeline; failure to reach the prediction rejects this proposed immediate benefit, despite the proven representational constraint.

Codex session ID: 01a164db-6519-7bd0-9379-aa4cfe2012a7
Resume in Codex: codex resume 01a164db-6519-7bd0-9379-aa4cfe2012a7
