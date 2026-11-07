**Two blockers found. Do not launch the full matrix yet.** Review was read-only; I ran no experiments and changed no files.

1. **BLOCKER — Finetuning fails when deep-copying the pretrained encoder.**  
   [llm_stem/run.py:170](<G:/broswer download/STEM-GNN/STEM-GNN/scripts/llm_stem/run.py:170>), [encoder.py:460](<G:/broswer download/STEM-GNN/STEM-GNN/model/encoder.py:460>)

   Training `Encoder.encode()` stores the differentiable, non-leaf tensor `env_reg_total / env_layers` in `_last_env_reg`, even though the router regularizer is excluded from the loss. `pretrain()` ends with `.eval()`, which does **not** clear that attribute. The subsequent `copy.deepcopy(enc)` encounters this non-leaf tensor and raises PyTorch’s unsupported tensor-deepcopy error.

   This affects both `run_cell()` and `select_coeff()`. The reported self-check bypasses this path: it constructs `RouterCodeModel(enc, vq, ...)` directly. Clear the unused cached regularizer before copying, or construct fresh modules and load their states.

2. **BLOCKER for the controlled H60–H62 interpretation — Shared components do not initialize identically across arms.**  
   [llm_stem/run.py:98](<G:/broswer download/STEM-GNN/STEM-GNN/scripts/llm_stem/run.py:98>), [encoder.py:214](<G:/broswer download/STEM-GNN/STEM-GNN/model/encoder.py:214>), [encoder.py:260](<G:/broswer download/STEM-GNN/STEM-GNN/model/encoder.py:260>)

   Identical seeds do not repair dimension-dependent RNG consumption. Router construction is interleaved with expert construction; then `reset_parameters()` consumes the already-diverged stream. Consequently, changing router dimensions changes **shared expert weights**, followed by the reconstruction decoders and VQ initialization.

   This directly affects S-R versus S-R0 and S0, and S-R versus S-U. S-R/S-RF/S-PQ have matching dimensions and avoid this particular mismatch. Finetuning’s separate reseed aligns the classifier initialization but cannot undo different pretrained starting points.

   Initialize shared components from identical saved states, or isolate router initialization from the shared RNG stream. The existing forward-equality checks do not test cross-arm initialization.

3. **MEDIUM — The weighted relaxation permits very large edge gradients near isolation.**  
   [llm_stem/run.py:125](<G:/broswer download/STEM-GNN/STEM-GNN/scripts/llm_stem/run.py:125>)

   `clamp_min(1e-6)` prevents division by zero; it does not bound gradients usefully. When incoming total weight approaches zero, derivatives can scale like `1 / total`; below the clamp, the numerator derivative still carries a factor of `1e6`. An isolated node with zero-weight candidate edges can exhibit this during discrete candidate scoring, although scoring itself uses no gradients.

   This is a **conditional numerical/attack-quality risk**, not a demonstrated NaN or inevitable failure. Binary-forward equality does not establish stability of the continuous attack trajectory. There are no finite-gradient checks or diagnostics here.

The remaining requested checks:

- **S-U freezing is correct.** Both router weights and biases are zeroed and have `requires_grad=False`. Both optimizers filter frozen parameters, so weight decay cannot move them. `CodeModel` only changes VQ parameter flags; it does not unfreeze routers. Parameter deepcopy preserves `requires_grad`. Training remains stochastic through Gumbel noise, exactly as Amendment 1 requires; evaluation routing is uniform.

- **State restoration is correct once construction succeeds.** The cloned `state_dict` includes the registered `rf` buffer, codebook buffers, and BatchNorm running mean, variance, and batch counter. Loading it preserves existing parameter freeze flags. Attacks run under `.eval()`, so BatchNorm statistics and dropout remain frozen.

- **PRBCD interface looks compatible.** The contrib source calls the model with three positional arguments, matching inherited `CodeModel.forward(x, ei, w=None)`. It accepts `tanh_margin`, requires `block_size > budget`, and defaults to undirected perturbations with symmetric adjacency construction. `100_000` is valid for these Cora budgets. **Version limitation:** the exact 2.8.0 source URL was unavailable; I checked the accessible [official contrib source](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/contrib/nn/models/rbcd_attack.html), rather than the cluster’s installed package.

- **Union evaluation is correct** at [run.py:246](<G:/broswer download/STEM-GNN/STEM-GNN/scripts/llm_stem/run.py:246>). Every arm sees every generated candidate; clean and lower budgets are included. `worst_node` intersects correctness per node, while `worst_graph` takes minimum graph accuracy. `own_worst_node` correctly restricts candidates to that arm. Building `edits` after attacks is valid and runs every validation before evaluation; it merely delays failure detection.

- **`route_tv_own` is specifically the highest-budget own `ste` graph**, not the worst candidate or an aggregate across attacks ([run.py:263](<G:/broswer download/STEM-GNN/STEM-GNN/scripts/llm_stem/run.py:263>)). Its implementation is correct for that diagnostic; describe that scope when reporting it.

- **Coefficient selection matches Amendment 1.** Seed zero gives `base=split`; all three splits are used. Selection reads clean validation scores, and chooses the largest coefficient within 0.005 of the best mean. No test accuracy enters selection.

- **No additional test-label supervision found.** Training uses training labels; checkpoint selection uses validation labels. Test labels enter the declared attacker and reporting. `y.max()` supplies class count. No text-derived adjacency, scalar edge weighting, or sampling path was found; text conditions the allowed expert transforms.

- **CPU execution:** attacks are sequential, totaling **62 PRBCD calls per cell**, including two surrogate attacks. The weighted implementation materializes edge-by-feature tensors for roughly twice the candidate block size; individual first-layer float32 intermediates can approach 300 MB for 384-dimensional inputs, with several intermediates and gradients live. Peak memory and wall time remain unverified. CPU `index_add_` is not the usual CUDA deterministic-algorithm problem; `warn_only=True` also does not guarantee reproducibility.

- **`trm[s]` indexing matches the repository’s split-major Cora convention**, corroborated by [task_constructor.py:60](<G:/broswer download/STEM-GNN/STEM-GNN/dataset/task_constructor.py:60>). I found no transpose bug. The actual serialized cluster masks were not inspected.
