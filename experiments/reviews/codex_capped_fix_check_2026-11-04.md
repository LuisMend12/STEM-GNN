- `STEM-GNN/scripts/capped/run.py:76`: BPDA’s forward can lose the capped value through cancellation (`g=1e8, B=1` in float32 produces 0 instead of 1). Fix: `capped.detach() + (g - g.detach())`, preserving the capped forward and identity gradient.

- `STEM-GNN/scripts/capped/run.py:110,118,225`: Only training loss is checked for finiteness. Nonfinite validation/test logits can silently become predictions through `argmax`, yielding plausible accuracy. Fix: check all logits for finiteness before loss, accuracy, certification, and attack evaluation.

- `STEM-GNN/scripts/capped/run.py:341`: Resume configuration omits Python/PyTorch and other dependency versions; PyTorch is recorded only afterward. Changed environments can silently reuse old checkpoints and attacks. Fix: include relevant versions and thread settings in the hashed, resume-validated configuration.

Static review only: the fresh-process selfcheck could not run because local PyTorch import failed under the read-only environment.