#!/bin/bash
#SBATCH --job-name=stem-gnn-pubmed-moe
#SBATCH --partition=general-gpu
#SBATCH --account=jhf24001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lam23005/STEM-GNN/logs/pubmed_moe_%j.out
#SBATCH --error=/home/lam23005/STEM-GNN/logs/pubmed_moe_%j.err

source ~/miniconda3/bin/activate STEM-GNN
cd /home/lam23005/STEM-GNN

PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
  --use_params --finetune_dataset pubmed --gpu 0 --debug --repeat 10 \
  --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --pretrain_dataset na

echo "====== DONE ======"
