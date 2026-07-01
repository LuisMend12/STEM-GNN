#!/bin/bash
#SBATCH --job-name=stem-ood-baselines
#SBATCH --partition=general-gpu
#SBATCH --account=jhf24001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lam23005/STEM-GNN/logs/ood_baselines_%j.out
#SBATCH --error=/home/lam23005/STEM-GNN/logs/ood_baselines_%j.err

source ~/miniconda3/bin/activate STEM-GNN
cd /home/lam23005/STEM-GNN

# Plain GNN (no VQ, no MoE) -- general GNN baseline
for DATASET in cora pubmed arxiv wikics; do
    echo "=== Plain GNN (no VQ, no MoE): $DATASET ==="
    PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
        --use_params --finetune_dataset $DATASET --gpu 0 --debug --repeat 10 \
        --pretrain_dataset na --use_vq 0
done

# GNN + VQ (no MoE) -- standard STEM-GNN without experts
for DATASET in cora pubmed arxiv wikics; do
    echo "=== GNN + VQ (no MoE): $DATASET ==="
    PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
        --use_params --finetune_dataset $DATASET --gpu 0 --debug --repeat 10 \
        --pretrain_dataset na
done

# GNN + VQ + MoE + lamda_env=0.1 (no pretrain, best no-pretrain config)
for DATASET in cora pubmed arxiv wikics; do
    echo "=== GNN + VQ + MoE + lamda_env=0.1 (no pretrain): $DATASET ==="
    PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
        --use_params --finetune_dataset $DATASET --gpu 0 --debug --repeat 10 \
        --pretrain_dataset na \
        --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1
done

echo "====== ALL DONE ======"
