#!/bin/bash
#SBATCH --job-name=stem-gnn-moe-pretrain
#SBATCH --partition=general-gpu
#SBATCH --account=jhf24001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lam23005/STEM-GNN/logs/moe_pretrain_%j.out
#SBATCH --error=/home/lam23005/STEM-GNN/logs/moe_pretrain_%j.err

source ~/miniconda3/bin/activate STEM-GNN
cd /home/lam23005/STEM-GNN

python STEM-GNN/pretrain.py \
    --use_params \
    --pretrain_dataset all \
    --pretrain_epochs 50 \
    --pretrain_run_id moe_pretrain \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 \
    --gpu 0 \
    --debug

echo "====== DONE ======"
