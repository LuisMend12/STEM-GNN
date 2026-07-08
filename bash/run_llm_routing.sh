#!/bin/bash
#SBATCH --job-name=stem-llm-routing
#SBATCH --partition=general-gpu
#SBATCH --account=jhf24001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/lam23005/STEM-GNN/logs/llm_routing_%j.out
#SBATCH --error=/home/lam23005/STEM-GNN/logs/llm_routing_%j.err

source ~/miniconda3/bin/activate STEM-GNN
cd /home/lam23005/STEM-GNN

# LLM routing consistency loss sweep on Cora
# Compares: no LLM loss / LLM loss only / lamda_env only / both combined

echo "=== MoE + lamda_env=0.1 (no LLM routing, baseline) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1

echo "=== MoE + llm_routing_reg=0.1 only (no lamda_env) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --llm_routing_reg 0.1

echo "=== MoE + lamda_env=0.1 + llm_routing_reg=0.05 (combined) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1 --llm_routing_reg 0.05

echo "=== MoE + lamda_env=0.1 + llm_routing_reg=0.1 (combined) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1 --llm_routing_reg 0.1

echo "=== MoE + lamda_env=0.1 + llm_routing_reg=0.2 (combined) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1 --llm_routing_reg 0.2

echo "=== MoE + lamda_env=0.1 + llm_routing_reg=0.1, Pubmed ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset pubmed --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1 --llm_routing_reg 0.1

echo "====== ALL DONE ======"
