# STEM-GNN architecture review: what can be improved (DRAFT, pending Codex review)

**Scope.** A code-level read of `model/encoder.py`, `model/vq.py`, `model/pt_model.py`, `model/ft_model.py`, `pretrain.py`, `finetune.py`, `task/{node,link,graph}.py`, and the dataset generators under `dataset/data/`. Each finding is checked against what the data actually contains.

## Data facts that decide which issues matter

| Dataset family | Edge text (`edge_text_feat`) | Directed? |
|---|---|---|
| Cora, Pubmed, arxiv, wikics | **one constant string per dataset**, e.g. Cora `"feature edge. connected papers are cited together by other papers."` (`single_graph/Cora/gen_data.py:56`) | no: Cora and Pubmed edge sets are 100% reciprocal |
| WN18RR, FB15K237 | **one constant string**, `"feature edge. relation between two entities."` (`KG/gen_data.py:93`). The relation-specific text is commented out (lines 90–91). | **yes:** WN18RR train triples are 34% reciprocal |
| chemhiv, chempcba, chemblpre | **per-bond text**, one per edge (`chemmol/gen_data.py:69-84`), so it is **informative** | no (molecular bonds) |

## Findings (ranked by expected impact)

### F1 (HIGH). Edge features are mishandled on both encoder paths

**(a) The non-MoE path adds raw text embeddings to hidden states at every layer.**

- `MySAGEConv.message` (`encoder.py:139-142`) computes `ReLU(x_j + xe)`. Here `xe` is the raw 768-d sentence embedding of the edge text, with no projection, and the same `edge_attr` is passed to every layer (`encoder.py:446,450`).
- At layer 0, `x_j` and `xe` share the text-embedding space. At layer ≥ 1, `x_j` is a learned hidden state, and adding a fixed text vector mixes two unrelated coordinate systems.
- It only runs because `hidden_dim == input_dim == 768`. Any other width raises a shape error.
- On the node and KG datasets the edge text is a constant, so the term is a fixed, untrained shift in front of the ReLU. Its norm is comparable to `x_j`'s at layer 0.

**(b) The MoE path ignores edge features entirely.** `MixtureSageLayer.forward(x, edge_index, edge_attr)` (`encoder.py:167-174`) never reads `edge_attr`.

- On molecules, bond-type information is silently discarded whenever `--moe` is on.
- In pretraining, `topo_sem_recon_loss` (`pt_model.py:68-83`) asks the encoder output to reconstruct edge-text embeddings that the MoE encoder never received.

**Proposed fix: learned per-layer edge projection (GINE-style).** Compute the message as `ReLU(x_j + W_e^ℓ · e_ij)` in both `MySAGEConv` and `MixtureSageLayer`, with one `W_e^ℓ` per layer. This makes edge information available to MoE, puts it in each layer's own space, and removes the `hidden_dim == input_dim` coupling.

### F2 (MEDIUM). The MoE and SAGE paths aggregate in opposite directions

- `MixtureSageLayer` aggregates `x[col]` into `row`, i.e. from `edge_index[1]` to `edge_index[0]` (`encoder.py:168-169`).
- `MySAGEConv` uses PyG's default `source_to_target` flow, from `edge_index[0]` to `edge_index[1]`.
- On undirected graphs the two are identical. On directed KGs (WN18RR is 34% reciprocal), switching `--moe` silently reverses which neighbors each node reads.
- **Fix:** use one convention, or aggregate both directions explicitly.

### F3 (MEDIUM). The MoE layer is not a drop-in SAGE expert

Compared with `MySAGEConv`, `MixtureSageLayer` (`encoder.py:154-174`) differs in several ways:

- no message nonlinearity;
- no bias;
- Xavier initialization of a stacked `[K, 2d, d]` tensor;
- a residual added **before** BatchNorm;
- no residual in the first layer.

`experiments/moe-depth` found a larger depth drop for MoE even with uniform routing (−3.85 vs SAGE −1.67; the difference is unresolved, SE 1.09). **Fix:** build each expert from the same operator as `MySAGEConv` (same message, bias and init), so MoE vs non-MoE differ only in routing.

### F4 (MEDIUM). The router regularizer is per-node entropy maximization, not load balancing

`_reg_loss` (`encoder.py:256-258`) computes `mean_n Σ_k w_nk log π_nk`. Minimizing it pushes *each node's* routing toward uniform, which works against expert specialization. Standard MoE balance losses act on *batch-level* expert usage instead. It is off by default (`lamda_env = 0`), so this only matters when it is enabled.

**Fix:** a batch-level importance/load loss, optionally with a per-node sharpness term.

### F5 (MEDIUM). The pretraining teacher runs in training mode

`sem_encoder` (the EMA teacher, `pt_model.py:22,93`) runs inside `model.train()`, so its targets carry dropout noise and Gumbel routing noise. BYOL/DINO-style teachers usually give deterministic targets. Separately, `ema_update_sem_encoder` (`pt_model.py:104-106`) averages parameters only; BatchNorm buffers are updated by the teacher's own forward passes.

**Fix:** run the teacher with dropout off and deterministic routing.

### F6 (MEDIUM). VQ behavior at finetune time

**(a)** The VQ commitment loss is discarded during node finetuning (`ft_model.py:142`, `task/node.py:74`). An unmatched comparison in `experiments/stem-vq-robustness` showed +4.1 points under attack with commitment. That is an observation, not a causal estimate.

**(b)** The straight-through estimator runs only in training (`vq.py:929-937`), so a white-box attack in eval mode sees zero gradients. This was confirmed on the STEM-lite proxy (≈12-point gap). It is an evaluation pitfall, not a training defect.

### F7 (LOW). Latent shape bug in pretraining

`topo_sem_recon_decoder = nn.Linear(hidden_dim*2, hidden_dim)` (`pretrain.py:123`) is compared by MSE against `edge_attr`, which has size `input_dim`. It works only because `hidden_dim == input_dim`. **Fix:** output `input_dim`.

## Evidence plan

- **F1(a)** is testable on local Cora with constant edge text. See `protocol.md`.
- **F1(b)** needs molecule data: the HuggingFace `molecule_property_instruction` dataset plus `rdkit`, neither installed. That needs the user's go-ahead.
- **F3** has a proposed matched-operator test in `../moe-depth/analysis.md`.
