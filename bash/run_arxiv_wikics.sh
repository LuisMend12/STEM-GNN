#!/bin/bash
#SBATCH --job-name=stem-arxiv-wikics
#SBATCH --partition=general-gpu
#SBATCH --account=jhf24001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lam23005/STEM-GNN/logs/arxiv_wikics_%j.out
#SBATCH --error=/home/lam23005/STEM-GNN/logs/arxiv_wikics_%j.err

source ~/miniconda3/bin/activate STEM-GNN
cd /home/lam23005/STEM-GNN

# --- ARXIV ---

echo "=== [Arxiv] MoE collapse test (no lamda_env) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset arxiv --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0

echo "=== [Arxiv] MoE + lamda_env=0.1 ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset arxiv --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1

echo "=== [Arxiv] MoE + lamda_env=0.1 + llm_routing_reg=0.1 ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset arxiv --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1 --llm_routing_reg 0.1

# --- WIKICS ---

echo "=== [WikiCS] MoE collapse test (no lamda_env) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset wikics --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0

echo "=== [WikiCS] MoE + lamda_env=0.1 ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset wikics --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1

echo "=== [WikiCS] MoE + lamda_env=0.1 + llm_routing_reg=0.1 ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset wikics --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1 --llm_routing_reg 0.1

echo "====== ALL DONE ======"
