#!/bin/bash
#SBATCH --job-name=stem-all-ood
#SBATCH --partition=general-gpu
#SBATCH --account=jhf24001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=/home/lam23005/STEM-GNN/logs/all_ood_scripts_%j.out
#SBATCH --error=/home/lam23005/STEM-GNN/logs/all_ood_scripts_%j.err

source ~/miniconda3/bin/activate STEM-GNN
cd /home/lam23005/STEM-GNN

# Run all 5 OOD scripts from the paper on Cora with best config (MoE + lamda_env, no pretrain)
# This answers: does the MoE+LLM benefit hold across all OOD types?

echo "=== 1/5 degree_shift_ood (Cora) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1

echo "=== 2/5 homophily_shift_ood (Cora) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/homophily_shift_ood.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1

echo "=== 3/5 missing_feature (Cora) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/missing_feature.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1

echo "=== 4/5 random_edge_drop (Cora) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/random_edge_drop.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1

echo "=== 5/5 tri_objective (Cora) ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/tri_objective.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 10 \
    --pretrain_dataset na \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1

echo "====== ALL DONE ======"
