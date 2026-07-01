#!/bin/bash
#SBATCH --job-name=stem-llm-ext
#SBATCH --partition=general-gpu
#SBATCH --account=jhf24001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=/home/lam23005/STEM-GNN/logs/llm_router_extended_%j.out
#SBATCH --error=/home/lam23005/STEM-GNN/logs/llm_router_extended_%j.err

source ~/miniconda3/bin/activate STEM-GNN
cd /home/lam23005/STEM-GNN

# Extended LLM-direct router comparison:
# 1. homophily_shift_ood on Cora/Pubmed/WikiCS — homophily is a structural property
#    so LLM routing (structure-invariant) should have clearest advantage here.
# 2. degree_shift_ood on Arxiv — larger graph, different domain.

echo "=== [BASELINE] GNN router, homophily_shift, Cora ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/homophily_shift_ood.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 5 \
    --pretrain_run_id lamda_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --lamda_env 0.1

echo "=== [LLM ROUTER] LLM-direct router, homophily_shift, Cora ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/homophily_shift_ood.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 5 \
    --pretrain_run_id lamda_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --lamda_env 0.1 --use_llm_router

echo "=== [BASELINE] GNN router, homophily_shift, Pubmed ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/homophily_shift_ood.py \
    --use_params --finetune_dataset pubmed --gpu 0 --debug --repeat 5 \
    --pretrain_run_id lamda_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --lamda_env 0.1

echo "=== [LLM ROUTER] LLM-direct router, homophily_shift, Pubmed ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/homophily_shift_ood.py \
    --use_params --finetune_dataset pubmed --gpu 0 --debug --repeat 5 \
    --pretrain_run_id lamda_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --lamda_env 0.1 --use_llm_router

echo "=== [BASELINE] GNN router, homophily_shift, WikiCS ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/homophily_shift_ood.py \
    --use_params --finetune_dataset wikics --gpu 0 --debug --repeat 5 \
    --pretrain_run_id lamda_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --lamda_env 0.1

echo "=== [LLM ROUTER] LLM-direct router, homophily_shift, WikiCS ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/homophily_shift_ood.py \
    --use_params --finetune_dataset wikics --gpu 0 --debug --repeat 5 \
    --pretrain_run_id lamda_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --lamda_env 0.1 --use_llm_router

echo "=== [BASELINE] GNN router, degree_shift, Arxiv ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset arxiv --gpu 0 --debug --repeat 3 \
    --pretrain_run_id lamda_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --lamda_env 0.1

echo "=== [LLM ROUTER] LLM-direct router, degree_shift, Arxiv ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset arxiv --gpu 0 --debug --repeat 3 \
    --pretrain_run_id lamda_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --lamda_env 0.1 --use_llm_router

echo "All extended LLM-direct router experiments done."
