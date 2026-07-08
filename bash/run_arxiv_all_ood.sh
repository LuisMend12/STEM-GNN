#!/bin/bash
#SBATCH --job-name=stem-arxiv-ood
#SBATCH --partition=general-gpu
#SBATCH --account=jhf24001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lam23005/STEM-GNN/logs/arxiv_full_ood_%j.out
#SBATCH --error=/home/lam23005/STEM-GNN/logs/arxiv_full_ood_%j.err

source ~/miniconda3/bin/activate STEM-GNN
cd /home/lam23005/STEM-GNN

# All 5 OOD scripts from the paper on Arxiv — MoE + lamda_env=0.1 (no pretrain)
# Arxiv is large (169K nodes), fewer epochs than cora/pubmed

echo "=== 1/5 degree_shift_ood (Arxiv) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset arxiv --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1

echo "=== 2/5 homophily_shift_ood (Arxiv) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/homophily_shift_ood.py \
    --use_params --finetune_dataset arxiv --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1

echo "=== 3/5 missing_feature (Arxiv) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/missing_feature.py \
    --use_params --finetune_dataset arxiv --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1

echo "=== 4/5 random_edge_drop (Arxiv) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/random_edge_drop.py \
    --use_params --finetune_dataset arxiv --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1

echo "=== 5/5 tri_objective (Arxiv) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/tri_objective.py \
    --use_params --finetune_dataset arxiv --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1

echo "====== ALL DONE ======"
