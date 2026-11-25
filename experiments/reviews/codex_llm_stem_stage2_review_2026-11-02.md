**Do not launch the full matrix yet. I found two definite code blockers and several design issues to resolve before freezing Stage 2.** This was a read-only review; no training, selfchecks, or experiments were run.

Paths below are relative to the workspace root.

1. **BLOCKER — S-V’s predictor has the wrong input width.**  
   `STEM-GNN/scripts/llm_stem/stage2.py:88,102`; `STEM-GNN/scripts/stem_fixes/run.py:168`; `STEM-GNN/model/vq.py:734–739`.

   `make_vq()` sets `codebook_dim=128, heads=4`, so `vq.project_in(z)` is `[N,512]`. The predictor is `Linear(128,1536)`. S-V crashes on its first semantic-loss forward; S-VP and S-VA cannot be produced either. Because `train_all()` constructs all arms regardless of `--arms`, restricting attacks does not bypass this.

   **Fix:** construct the predictor with the actual projected dimension, 512 here, and assert its input shape. Preserve the existing VQ head layout; changing `codebook_dim` to 32 would change S0’s architecture.

2. **BLOCKER — `audit()` assumes a tensor, but the actual masks are lists.**  
   `stage2.py:296`; `stem_fixes/run.py:50–53`.

   I inspected the Cora archive’s pickle metadata without loading model code: `val_masks` is a list of Boolean `[2708]` tensors. Consequently, `vam.size()` raises `AttributeError`; the suspicious axis expression is never reached.

   **Fix:** explicitly stack the intended split masks, assert `[splits,N]` Boolean shape, then reduce over dimension 0. Specify which splits are eligible: silently using all cached splits would include splits outside the declared `{0,1,2}` study. Assert a nonempty audit subset and finite logits for **every rotation**, rather than checking availability from rotation 0 alone.

3. **HIGH — The audit decision is disconnected from the actual prior selection.**  
   `stage2.py:287–301,347,360`; `experiments/llm-stem/protocol.md:169`.

   The audit merely prints `use_three_rotations`; subsequent jobs independently default to `--prior_perms 0`. Nothing enforces the prescribed `{0,2,4}` decision or records the audit evidence in each result. This permits a full matrix using the wrong prior.

   **Fix:** persist a frozen audit manifest and require cell jobs to consume and validate it. Define whether rotation selection uses a global validation pool or split-specific validation; a union across splits can include nodes that are test nodes in another split and needs explicit disclosure.

   The perm-0 filename formatting itself is correct. However, this workspace contains only `perm0.partial.pt` and its backup, not the final filename expected by the loader; the audit files are also absent locally. Cluster availability needs verification before submission.

4. **HIGH — S-F changes graph corruptions and shared initialization through RNG consumption.**  
   `stage2.py:81–98`; `llm_stem/run.py:82–99`.

   Creating the differently sized feature decoder consumes a different number of random draws before topology decoder/VQ initialization. Each epoch then calls feature masking before edge dropout: the default mask samples feature columns, so 1,536-dimensional Qwen input consumes a different RNG sequence from MiniLM. Graph dropout and subsequent encoder randomness therefore differ despite equal seeds. [PyG’s implementation confirms the column-wise sampling](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/utils/augmentation.html).

   This does not automatically invalidate a feature-replacement comparison, but it contradicts strict “only the declared intervention differs” pairing and adds avoidable noise to H70/H73.

   **Fix:** give shared modules independent initialization seeds and isolate feature masking, graph dropout, negative sampling, and encoder stochasticity into separate RNG streams. Input-dependent first-layer parameter counts remain a declared limitation.

5. **HIGH — MIX’s attack gradients can weaken substantially as alpha increases.**  
   `stage2.py:161–163,237–247`; `llm_stem/run.py:205–216`.

   MIX’s edge sensitivity carries a factor of `1-alpha`. With fixed PRBCD settings, probability-margin optimization can consequently make smaller steps and find weaker attacks. This mixes legitimate reliance on an edge-independent teacher with an attack-optimization disadvantage. The shared union helps but does not establish attack adequacy.

   **Fix:** prespecify a gradient-scale/step-size sensitivity check or an appropriately rescaled MIX attack objective, alongside convergence and discrete-search checks. Keep all resulting candidates in the common union. Until then, restrict conclusions to performance on this finite suite.

   There is **no output-format bug**: PyG’s probability margin applies softmax, and its tanh margin applies log-softmax, so normalized log-probabilities are accepted. [Official PRBCD source](https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/contrib/nn/models/rbcd_attack.html). The `attack_mode` property delegates correctly through `hasattr`, the `(acc,-alpha)` maximum chooses the smallest tied alpha, and skipping alpha 1 attacks is appropriate.

6. **MEDIUM — Low-label MIX’s selected alpha is missing from the results.**  
   `stage2.py:263,269–272`; protocol staged comparison at `protocol.md:189`.

   Standard and low-label MIX independently choose alpha, but rows record only the standard alpha. The low-label MIX endpoint is computed correctly; its selection cannot be reconstructed from the saved output.

   **Fix:** record `low_alpha`, low-label checkpoint metadata, the selected training node IDs, and the frozen prior/audit provenance. This makes H73–H75’s staged comparisons reviewable.

7. **MEDIUM — The selfcheck can report success without validating the claimed mechanism.**  
   `stage2.py:304–337`.

   Most checks print booleans rather than assert them. Comparing encoder weights after five epochs does not isolate the semantic-loss gradient. The MIX check differentiates the sum of log-probabilities through default STE; this is neither the actual attack objective nor a check of every backward mode. A failed check still exits successfully.

   **Fix:** make required checks assertions. Directly differentiate the semantic loss with respect to `z`, `project_in.weight`, and encoder weights; check finite nonzero gradients. Verify MIX mode propagation, hard-forward equivalence, and the actual attack objectives across modes. `enc.layers.parameters()` and `torch.equal(a,b)` are valid as written—the problem is what the comparison proves.

8. **MEDIUM — Statistical rules do not fully cover the staged integration claims.**  
   `experiments/llm-stem/protocol.md:173–190`.

   The two Holm families for H70–H75 are explicit and reasonable. But testing staged contrasts at ordinary 95% after selecting supported placements/endpoints leaves multiplicity across those selected claims unspecified. “Clean change ≥ −1” also specifies a point-estimate guardrail, not inferential preservation of clean accuracy.

   **Fix:** freeze test direction, exact nulls, family handling for staged claims, and the clean noninferiority rule. Clarify that a +2 point estimate with significance against zero does not establish a true effect exceeding +2.

   The per-cell maximum of graph-free baselines is a defensible, conservative **oracle benchmark** for demonstrating superiority. Failure to beat it does not establish that a particular deployable graph-free model is better. Replace the protocol’s categorical “better choice” conclusion with that narrower interpretation.

9. **MEDIUM — Runtime and recovery risk are substantial; there is no demonstrated concurrency bug.**  
   `stage2.py:237–283,369–372`; `llm_stem/run.py:212`.

   The default cell makes 82 PRBCD calls, or 1,230 across 15 cells, unless alpha 1 removes MIX attacks. It then evaluates 11 models across those candidates. Attacks execute sequentially; eight threads do not mean eight simultaneous attacks. A 100,000-candidate undirected block expands the message tensors substantially, particularly for S-F’s first layer.

   Results are written only after all attacks, low-label training, and evaluation. A late failure loses the cell’s completed work, and fixed filenames allow reruns to overwrite results.

   **Fix:** save resumable model/candidate checkpoints and use configuration-specific output names. Establish resource requirements before the full matrix; static review cannot certify RAM or wall time.

The remaining requested paths look correct, subject to fixing the predictor:

- **S-V gradients and VQ state:** `project_in` is a differentiable linear layer. Its second invocation reaches the encoder and does not update EMA or BatchNorm. The preceding real `vq(z)` updates codebook buffers once during pretraining; it does not mutate `z` or the projection weights before that second call. This is representation distillation at the quantizer input; a VQ-specific benefit still requires the promised no-VQ control.
- **Anchor:** `eval()` retains autograd. The extra forward uses soft routing and running BatchNorm statistics, with dropout disabled; it consumes no router/dropout RNG and makes no additional BN update. `kl_div(log_model, prior, batchmean)` computes `KL(prior || model)`, summed over classes and averaged over selected nodes. `~(tr|va)` excludes both masks. Validation-calibrated pseudo-label supervision is declared; I found no direct test-label use in training/calibration.
- **Isolation and reuse:** finetuning deep-copies pretrained encoders and quantizers. Standard/low-label reuse is appropriate. MIX deliberately shares S0, but attacks run in evaluation mode and restore `attack_mode`; the wrapper does not call real VQ during finetuning or attacks, so EMA cannot drift.
- **Baselines and outputs:** `fx.fit` receives the correct arguments, restores the best state, and returns `(best,epochs)`; ignoring that return is harmless. `model(x)` is correct for MLP. `cr.correct` gives constant graph-free accuracy. MLP uses the same Adam settings, width, dropout, patience, and epoch cap, though it has less capacity and no pretraining or STEM penalty. This supports comparison with the **specified** MLP, not an optimally tuned graph-free model.
- **Masks, rows, serialization:** individual `trm[s]` masks are Boolean, so low-label masking and indexing are correct. Row values are Python scalars, feature tensors are excluded, and graph keys serialize consistently. Evaluating the argument to `fx.acc` outside `no_grad` briefly creates an unnecessary graph but does not retain it or corrupt results.
- **Edge constraint:** no reviewed path uses text to select, add, drop, or assign scalar weights to edges under Amendment 1’s operational definition. Topology reconstruction remains an auxiliary loss.