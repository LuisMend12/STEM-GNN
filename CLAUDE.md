# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment Setup

```bash
conda env create -f environment.yml
conda activate STEM-GNN
```

Key dependencies: Python 3.9, PyTorch 1.13.1 + CUDA 11.6, PyG 2.3.0, DGL 1.0, pytorch-lightning 2.0.4, wandb.

## Running

All commands are run from the repo root (`STEM-GNN/`). The inner `STEM-GNN/` directory is the Python package root — scripts import from `model/`, `dataset/`, `utils/`, and `task/` as top-level packages, so you must run from the outer root (not from inside the package directory).

**Pretrain:**
```bash
python STEM-GNN/pretrain.py --use_params --gpu 0 --pretrain_dataset all --pretrain_epochs 50
```
Checkpoints are saved to `STEM-GNN/ckpts/pretrain_model/<run_id>/encoder_<epoch>.pt` and `vq_<epoch>.pt`.

**Finetune:**
```bash
python STEM-GNN/finetune.py --use_params --finetune_dataset cora --gpu 0
# Load a specific pretrain checkpoint epoch:
python STEM-GNN/finetune.py --use_params --finetune_dataset cora --pretrain_model_epoch 25
# Point to a checkpoint folder directly:
python STEM-GNN/finetune.py --use_params --finetune_dataset cora --pretrain_path STEM-GNN/ckpts/pretrain_model/<run_id>
```

**OOD / variant scripts (node tasks only):**
```bash
python STEM-GNN/scripts/degree_shift_ood.py --use_params --finetune_dataset cora
python STEM-GNN/scripts/homophily_shift_ood.py --use_params --finetune_dataset cora
python STEM-GNN/scripts/missing_feature.py --use_params --finetune_dataset cora
python STEM-GNN/scripts/random_edge_drop.py --use_params --finetune_dataset cora
python STEM-GNN/scripts/tri_objective.py --use_params --finetune_dataset cora
```

Pass `--debug` to disable WandB logging (runs in `mode="disabled"`).

## Architecture Overview

STEM-GNN is a **pretrain → finetune** framework for text-attributed graphs. The core idea: pretrain a GNN encoder with vector-quantized discrete representations across multiple graph domains, then finetune a linear head on downstream tasks.

### Two-phase pipeline

**Pretraining (`pretrain.py` + `model/pt_model.py`)**

`PretrainModel` wraps:
- `Encoder` — multi-layer GNN (default: GraphSAGE with edge features)
- `VectorQuantize` — multi-head VQ codebook (adapted from `lucidrains/vector-quantize-pytorch`)
- A momentum-updated `sem_encoder` (EMA copy of the encoder used as a semantic teacher)
- Three reconstruction decoders: feature (`feat_recon_decoder`), topology (`InnerProductDecoder`), and topology-semantic (`topo_sem_recon_decoder`)

The pretrain loss combines: feature reconstruction + topology reconstruction + topology-semantic reconstruction + semantic (cosine similarity against the EMA encoder) + VQ commitment loss + optional MoE regularization.

Input data uses augmentation: random feature masking (`mask_feature`) and edge dropout (`dropout_adj`) on the augmented view; the original graph is the reconstruction target.

**Finetuning (`finetune.py` + `model/ft_model.py`)**

`TaskModel` wraps the pretrained `Encoder` + `VectorQuantize`. A linear `decoder` is trained on top of VQ codes. VQ parameters are frozen by default (`--freeze_vq 1`). Supports node classification, link prediction (edge-type classification), and graph classification via task-specific train/eval loops in `task/node.py`, `task/link.py`, `task/graph.py`.

### Encoder (`model/encoder.py`)

The `Encoder` supports multiple GNN backbones (`sage`, `gat`, `gcn`, `gin`). The optional **Mixture-of-Experts (MoE)** variant replaces standard SAGEConv layers with `MixtureSageLayer`, where a per-node router (via Gumbel-softmax during training, softmax at inference) selects a weighted mix of `K` expert weight matrices. `--moe_layers` controls which layers use MoE (`none`/`last`/`all`).

Edge features are incorporated in `MySAGEConv.message()` by adding `edge_attr` to neighbor features before aggregation.

### Data layer (`dataset/`)

Dataset loading is delegated to a `UnifiedTaskConstructor` (adapted from the [OneForAll](https://github.com/LechengKong/OneForAll) framework). Graphs are stored as OFA-format PyG datasets with `node_text_feat` and `edge_text_feat` as sentence-encoder embeddings (768-dim by default). Node/edge indices (`x`, `xe`) are stored as integer indices into those feature tables, not the raw features, to allow batching across datasets with different feature pools.

`get_pt_data()` merges multiple datasets into a single `Batch` object by offsetting `x` and `xe` indices per dataset. Dataset mixing weights are defined in `config/pt_data.yaml` and used for weighted node sampling during pretraining.

Raw data must be placed under `STEM-GNN/data/`. Processed datasets are cached by PyG automatically.

### Config system

`--use_params` loads defaults from `config/pretrain.yaml` or `config/finetune.yaml` (keyed by task then dataset name). CLI flags override config values. The pretrain run ID is derived from hyperparameters via `get_pretrain_run_id()`, so changing hyperparameters auto-generates a distinct checkpoint directory.

### Supported datasets

| Task | Datasets |
|------|----------|
| Node classification | cora, pubmed, arxiv, wikics |
| Link prediction | WN18RR, FB15K237 |
| Graph classification | chemhiv, chempcba |
| Pretrain-only | chemblpre |

`dataset2task` in `finetune.py` maps dataset name → task type and drives all task-specific dispatch.
