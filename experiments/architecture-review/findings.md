# STEM-GNN architecture review: what can be improved

Reviewed by Codex (`../reviews/codex_architecture_review_2026-10-22.md`). Corrections are applied, and every claim marked **[verified]** was re-checked by me against the code or data. Code paths are relative to `STEM-GNN/`.

## Data facts that decide which issues matter

| Dataset family | Edge text | Directed? |
|---|---|---|
| Cora, Pubmed, arxiv, wikics | one **constant** string per dataset (Cora: `dataset/data/single_graph/Cora/gen_data.py:57`) | no: Cora and Pubmed are 100% reciprocal **[verified]** |
| WN18RR, FB15K237 | one **constant** string (`KG/gen_data.py:93`); relation-specific text is commented out (lines 90–91) | **yes:** WN18RR train triples are 34.4% reciprocal **[verified]** |
| chemhiv, chempcba, chemblpre | **per-bond text**, which is informative (`chemmol/gen_data.py:69-84`) | no |

## Ranked improvement opportunities

### 1. The VQ codebook never learns to fit the data (HIGH, all pretraining) **[verified]**

Both VQ constructors pass `ema_update=False` and leave the outer `learnable_codebook=False` (`pretrain.py:105-118`; `model/vq.py:722`). Because `ortho_reg_weight=1` (the default), the inner codebook *is* an `nn.Parameter` (`vq.py:785`: `learnable_codebook = has_codebook_orthogonal_loss or learnable_codebook`). But two detaches cut it off from the data:

- the commitment target is detached (`vq.py:931`);
- the straight-through estimator detaches the reconstruction path (`vq.py:937`).

So the codebook's only gradient comes from the orthogonality regularizer (on at most 32 random codes per step) and AdamW weight decay. **STEM's "tokens" are effectively a random, near-orthogonal unit-norm dictionary; only the encoder adapts to it.** This may be intentional, as in a random-projection quantizer, but nothing documents it.

**Test:** backpropagate each loss separately and inspect codebook gradients; reconstruction and commitment should give zero. Then compare an EMA-updated or truly learnable codebook against the current one.

### 2. MoE experts are initialized about 512× too small (MEDIUM–HIGH, MoE) **[verified]**

`nn.init.xavier_uniform_` on the stacked `[K, 2d, d]` tensor (`encoder.py:161-165`) computes fan from the 3D shape, so the variance is about 2/[d(2d+K)] instead of 2/(3d) per matrix. At d=768 and K=3:

- measured variance ratio: **512.6×**;
- transform RMS **0.0018** vs residual RMS **0.036**.

Every equal-width MoE layer therefore starts as close to an identity map. This is a strong candidate explanation for the extra MoE depth drop in `../moe-depth`, which is untested.

**Fix:** initialize each expert matrix separately. **Test:** compare transform/residual RMS and gradients at initialization, then accuracy at depths 2 and 16.

### 3. KG relation classification cannot see edge direction (HIGH on FB15K237) **[verified]**

`task/link.py:7-8` classifies with `(z_h + z_t)/2`, so `(h,t)` and `(t,h)` always get the same prediction. Over all local triples, an oracle keyed by endpoint identity can reach:

| Dataset | Unordered pair | Ordered pair |
|---|---|---|
| FB15K237 | **84.81%** | **91.54%** |
| WN18RR | 99.80% | 99.87% |

These are in-sample identity-collision ceilings, not achievable test accuracies.

**Fix:** an ordered pair representation such as `[z_h, z_t]`. **Cheapest decisive test (proposed by Codex):** linear probes on frozen entity text embeddings, comparing `(x_h + x_t)/2` with `[x_h, x_t]`. The prediction is ≥ +2 points on FB15K237.

### 4. Edge features are mishandled on both encoder paths (mechanism verified; accuracy impact not shown)

- **SAGE:** adds *unprojected* edge embeddings before the message ReLU at every layer (`encoder.py:139-142,446,450`). With multiple layers, deeper additions need matching hidden and edge widths; a one-layer encoder can have a different output width.
- **MoE:** active MoE layers **ignore** edge attributes (`encoder.py:167-174`). Hybrid encoders can still read them through their SAGE layers, and `--moe` alone does not activate experts when `moe_layers='none'`.
- **E1 (Cora, constant edge text; see `analysis.md`):** the raw injection did not cross the pre-registered harm threshold, and a learned per-layer projection did not improve the point estimates.
- **The fix still matters for compatibility and information access.** Its accuracy value needs informative edges (molecules), which are not available locally.

Note on reconstruction targets: asking an encoder to reconstruct edge attributes it never received is not contradictory in itself, since it can infer them from the endpoints. The stronger argument for F1b is the lost ability to condition messages on bonds.

### 5. Directed KG edges are made undirected in pretraining augmentation (HIGH for KG pretraining)

`dropout_adj(..., force_undirected=True)` (`pretrain.py:42-44`) selects edges by endpoint ordering and mirrors the ones it keeps. On directed KGs this can drop one-way edges according to arbitrary entity IDs and invent reverse edges, while the reconstruction targets stay directed.

**Test:** run the augmentation on a one-way edge in each orientation, and at `edge_p=0`.

### 6. MoE and SAGE aggregate in opposite directions (MEDIUM for directed data) **[verified]**

- MoE reads `x[col]` into `row`, i.e. outgoing neighbors (`encoder.py:168-169`).
- Default-flow SAGE reads incoming neighbors.
- They agree on reciprocal graphs.

Standardize the direction before attributing any difference to routing.

### 7. The MoE layer is not a drop-in SAGE expert (MEDIUM)

It has no message ReLU, no bias, the init issue from #2, and a residual added *before* BatchNorm. The residual is present whenever input and output widths match, **including layer 0 at the default 768→768**; the 384→64 harness had none at layer 0.

A joint `[agg, x]W` is algebraically equivalent to separate neighbor and root maps, so it is not an expressivity difference.

### 8. The pretraining LR schedule cycles every 100 minibatches (MEDIUM) **[verified]**

`get_scheduler(..., epochs=pretrain_epochs=50)` (`utils/others.py:138-141`) is stepped after **every minibatch** (`pretrain.py:64-65`). The cosine lambda therefore repeats with a period of 100 steps instead of decaying once over training.

**Fix:** use a total-step horizon, or step once per epoch.

### 9. `normalize='layer'` silently uses BatchNorm, and the final BatchNorm feeds the VQ (MEDIUM)

Every norm is `BatchNorm1d` (`encoder.py:227`), whatever `normalize` says. The final representation is batch-normalized before VQ assignment, so the codes depend on batch statistics, and those differ between sampled and full neighborhoods and between dataset mixes.

### 10. The cosine codebook does not renormalize embeddings at lookup (MEDIUM)

Queries are L2-normalized, but lookup uses `self.embed` directly (`vq.py:648-650`). Combined with #1 (weight decay plus orthogonal-loss updates), code norms can drift and change which code wins.

### 11. The finetune decoder is one linear classifier, not an ensemble (MEDIUM)

Each head's output reads all concatenated codes (`ft_model.py:43-46,142-146`), so averaging logits across heads is algebraically one linear map. Also, `heads=4` with `codebook_dim = code_dim = 768` gives a **3072-dimensional** code space (four separate 768-d codebooks), not 768 split four ways.

### 12. The VQ loss is discarded in every finetune path (MEDIUM) **[verified for node]**

`ft_model.py:142` drops the VQ loss, and so do `task/node.py:74`, `task/link.py:36` and `task/graph.py:36`. That loss includes commitment and the orthogonal regularizer. The exploratory +4.1 robustness observation came from the STEM-lite proxy.

### 13. The pretraining teacher runs in training mode (MEDIUM; harm is a hypothesis)

The EMA teacher uses dropout, Gumbel routing and training-mode BatchNorm. EMA updates parameters, not buffers. A deterministic teacher needs its BatchNorm buffers handled explicitly; simply calling `.eval()` would leave the teacher's statistics stale.

### 14. Other issues (LOW–MEDIUM)

- **Pretraining-loss quality:**
  - Constant edge targets allow a decoder-bias solution.
  - Topology negatives are drawn after positives are subsampled, so omitted true edges can become negatives.
  - Feature reconstruction covers unmasked features too.
  - Loss weights are not gradient-balanced.
- **Router regularizer:** it is a stochastic per-node term, `mean Σ w_gumbel·log_softmax(logits)`, not batch load balancing, and it is off by default.
- **Finetuning reports best-validation metrics but never restores those weights,** which only affects post-run callbacks (`utils/early_stop.py`, `finetune.py`).
- **KG evaluation is transductive:** held-out edges take part in message passing (`task/link.py:20-29`). Restoring relation-specific edge text would leak the labels.
- **Latent shape bugs that default widths hide:**
  - the edge-semantic decoder outputs `hidden_dim` where it should output `edge_dim` (`pretrain.py:123`);
  - the shared-decoder branch assumes `code_dim == hidden_dim` (`ft_model.py:46`).
- **Eval-mode VQ disconnects input-to-code gradients** (`vq.py:929-937`). A white-box attack gets either zero gradients or a disconnected-gradient error. Use an attack-only STE/BPDA wrapper.
