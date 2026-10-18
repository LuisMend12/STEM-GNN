---
name: robustness-experiment
description: Design, run, and record a GNN robustness experiment in STEM-GNN with a written protocol, sanity checks, adaptive attacks, and an honest results log. Use this whenever the user asks to test, probe, benchmark, or "collect evidence" for a robustness idea (edge reweighting, Lipschitz or sensitivity penalties, trust gates or budgets, VQ or MoE stability, OOD or shift scripts, missing features, edge drop), or asks whether some change "increases robustness", even if they do not say "experiment".
---

# Robustness experiment loop (STEM-GNN)

This skill adapts the inner and outer loops of Orchestra Research's `autoresearch` skill to this repo and its rules. The main changes from the original:

- The user is not assumed to be away. Ask before long or costly runs, commits, and anything outward-facing.
- There is no mandatory `/loop`. Use background runs and their completion notifications instead.

The evaluation rules come from `reports/Recent GNN robustness methods.md`.

## 0. Workspace

| What | Where |
|---|---|
| Experiment code | `STEM-GNN/scripts/<experiment_slug>/` |
| Protocol and analysis | `experiments/<slug>/protocol.md` and `experiments/<slug>/analysis.md` at the repo root |
| Raw results (json/csv/logs) | `experiments/<slug>/results/` |
| Running synthesis | `experiments/findings.md`, kept across experiments |

Reuse the existing harness when it fits. `STEM-GNN/scripts/adaptive_robustness/` already provides MLP, GCN, SAGE, the Q2 regularizers, text-gate and trust-budget variants, and RUNG-lite. It runs random-insertion, camouflage and PRBCD attacks on OFA Cora with frozen 384-d text embeddings.

**Environment facts** (check them again; they may change):

- The `STEM-GNN` conda env is not installed. Use `C:\Users\Luis Mendez\anaconda3\python.exe`, which has CPU-only torch and PyG 2.8 including `torch_geometric.contrib` PRBCD.
- The GPU is an RTX 3060 with 6 GB, but no CUDA torch is installed.
- Raw OFA files exist under `STEM-GNN/dataset/data/single_graph/`. There are no pretrain checkpoints, so the full STEM-GNN finetune pipeline cannot run yet. Say so rather than silently substituting a different model.

## 1. Protocol first (before any result exists)

Write `protocol.md` with these sections:

- **Hypothesis:** mechanism → prediction. For example: "per-node trust budget caps the neighbor share, so PRBCD needs more edges to flip a node; predict a smaller drop than SAGE at 10%."
- **Variants and the one thing each changes:** include a baseline that differs *only* in that change.
- **Threats** (choose deliberately):
  - **Random corruption:** edge drop/add, feature masking, degree/homophily shift. These test corruption robustness only.
  - **Adaptive white-box attack** (PRBCD/PGD through the model's own edge weighting). This is required before any word like "adversarially robust" appears.
  - **Defense-aware attack** on any cosine or semantic gate: camouflage (PGD-Guard-style), meaning edges to high-cosine wrong-label nodes.
  - **Poisoning versus evasion:** state which one.
- **Metrics:** clean accuracy plus accuracy under each threat. Report mean±std over ≥3 splits or seeds, and runtime.
- **Decision rule:** what result counts as support and what counts as refutation. Write it before running.
- **Confirmatory versus exploratory:** anything added after seeing results is exploratory.

If the user agrees, commit the protocol before running (`research(protocol): <slug>`). Never put the protocol and its results in one commit.

## 2. Mandatory checks (the reasons are in the brief)

1. **MLP reference.** A structure attack cannot hurt a structure-free model. If a "robust" GNN ends up near MLP accuracy, it has learned to ignore the graph. That is not a win unless its clean accuracy stays above MLP.
2. **Clean-accuracy floor.** Report the clean cost of every defense next to its robust gain.
3. **Gradient masking.** If an adaptive attack barely moves a model:
   - check that gradients reach `edge_weight` (they are non-zero for some candidate edges);
   - try a transfer attack from an undefended surrogate;
   - try the camouflage attack.

   Non-differentiable parts (VQ quantization, hard thresholds, argmax routing) need a straight-through or BPDA surrogate, or a transfer attack.
4. **Same text encoder** for every baseline that uses text similarity, because the encoder alone changes rankings (TGRB).
5. **Tune per variant on validation only**, on the clean validation set, then freeze the setting before attacking. Record the grid.
6. **Sanity checks** before trusting numbers: the run converged, the baseline reproduces (SAGE about 80% on OFA Cora split 0), and attack budgets are counted in undirected edges.

## 3. Running

- Run long jobs in the background (`run_in_background`), in parallel by split if the CPU allows, with per-split output files. Don't poll; wait for the notification.
- Smoke-test one model, split and rate first, and time it. Then estimate the total time and tell the user before launching anything that takes more than about 30 minutes.

## 4. Record and interpret (outer loop)

Write `analysis.md` with:

- the result table (mean±std);
- which hypotheses held and which failed, with the *mechanism* explaining each;
- what each negative result rules out;
- threats to validity (single dataset, CPU-scale PRBCD, simplified re-implementations such as "RUNG-lite", which must not be presented as RUNG).

Then update `experiments/findings.md`. It has four sections: current understanding, patterns, lessons/constraints (for example "trust_budget kappa untuned"), and open questions.

Choose the next step:

- **Deepen:** an ablation or a sub-hypothesis.
- **Broaden:** another dataset or threat.
- **Pivot:** if an assumption broke.
- **Conclude:** when one claim is supported under adaptive attack with ablations.

Before claiming anything externally, run the `research-rigor-review` skill.

## Wording rules for claims

- "Robust" without a qualifier means robust under adaptive attack. Otherwise write "robust to random edge insertion", and so on.
- Never call a re-implementation by the paper's name without "-lite" or "-style".
- A gain within one std of the baseline is "no detectable difference", not "slight improvement".
