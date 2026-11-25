# Project Map

## Purpose
STEM-GNN: pretrain a GNN encoder (MoE + vector quantization) on text-attributed graphs, finetune a
linear head downstream. Current research work: find evidence-backed ways to improve its
robustness/generalization, specifically structure-only VGAE neighbor weighting.

## Requirements
- Defense edge weights must never be computed from text (structure only); the attacker using
  the model loss is the threat model and is exempt. (User constraint.)
- Every design and results write-up is Codex-reviewed, and its fixes applied, before being shown. (User requirement.)
- All runs, checks and analysis happen on the UConn HPC cluster, not locally. (User requirement.)
- Pre-registered hypotheses with operational support/against rules (mean ≥ +2 and Holm p < .05;
  upper CI < +1), conditional on fixed graphs and splits.

## Components
- Core package: STEM-GNN/ (model/, dataset/, task/, pretrain.py, finetune.py)
- STEM-lite experiment scripts: STEM-GNN/scripts/ (llm_stem/stage2.py recipe, llm_stem/run.py
  PRBCD suite, adaptive_robustness/models.py baselines incl. RUNG-lite, code_robustness/run.py
  validation helpers)
- First VGAE experiment (implemented, finished): STEM-GNN/scripts/vgae_weighting/run.py,
  analyze.py; experiments/vgae-weighting/
- Follow-up (implemented by AI, not yet run): STEM-GNN/scripts/vgae_weighting/followup.py,
  analyze_followup.py; experiments/vgae-followup/protocol.md

## Main Flow
data (Cora/Pubmed OFA .pt) → STEM pretrain → structure-only VGAE scorer → neighbor weights
→ finetune arms → Part R (random injection, poisoning) / Part A (adaptive attacks, evasion)
→ per-cell JSONs on cluster → analysis (Holm family) → Codex results review → write-up

## Data and Trust Boundaries
- Datasets on cluster: STEM-GNN/dataset/data/single_graph/{Cora,Pubmed}
- Results: experiments/*/results (cluster `~/stem-llm`, copied back)

## Build and Deployment
- Cluster: Slurm, general partition (12h max), conda env stem-llm; sbatch scripts in ~/stem-llm

## Unknowns
- Whether the follow-up's attack suite is strong enough to rule out attack weakness
- Pubmed compute cost per cell (pending smoke timing)
