#!/bin/bash
#SBATCH --job-name=stem-llm-router
#SBATCH --partition=general-gpu
#SBATCH --account=jhf24001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/lam23005/STEM-GNN/logs/llm_direct_router_%j.out
#SBATCH --error=/home/lam23005/STEM-GNN/logs/llm_direct_router_%j.err

source ~/miniconda3/bin/activate STEM-GNN
cd /home/lam23005/STEM-GNN

# Compare GNN router vs LLM-direct router on degree-shift OOD.
# LLM-direct router: routes using raw 768-dim LLM text features (structure-invariant)
# instead of GNN hidden states — key claim for paper.
# Higher class_purity + lower routing shift = better OOD robustness via semantic routing.

echo "=== [BASELINE] GNN router, lamda_pretrain, Cora ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 5 \
    --pretrain_run_id lamda_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --lamda_env 0.1

echo "=== [LLM ROUTER] LLM-direct router, lamda_pretrain, Cora ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 5 \
    --pretrain_run_id lamda_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --lamda_env 0.1 --use_llm_router

echo "=== [BASELINE] GNN router, lamda_pretrain, Pubmed ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset pubmed --gpu 0 --debug --repeat 5 \
    --pretrain_run_id lamda_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --lamda_env 0.1

echo "=== [LLM ROUTER] LLM-direct router, lamda_pretrain, Pubmed ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset pubmed --gpu 0 --debug --repeat 5 \
    --pretrain_run_id lamda_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --lamda_env 0.1 --use_llm_router

echo "=== [BASELINE] GNN router, lamda_pretrain, WikiCS ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset wikics --gpu 0 --debug --repeat 5 \
    --pretrain_run_id lamda_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --lamda_env 0.1

echo "=== [LLM ROUTER] LLM-direct router, lamda_pretrain, WikiCS ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset wikics --gpu 0 --debug --repeat 5 \
    --pretrain_run_id lamda_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --lamda_env 0.1 --use_llm_router

echo "All LLM-direct router experiments done."
