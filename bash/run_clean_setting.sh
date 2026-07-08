#!/bin/bash
#SBATCH --job-name=stem-clean-moe
#SBATCH --partition=general-gpu
#SBATCH --account=jhf24001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/lam23005/STEM-GNN/logs/clean_setting_%j.out
#SBATCH --error=/home/lam23005/STEM-GNN/logs/clean_setting_%j.err

source ~/miniconda3/bin/activate STEM-GNN
cd /home/lam23005/STEM-GNN

echo "=== CLEAN SETTING: MoE-pretrained, Cora, no lamda_env ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/clean_setting_moe.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 10 \
    --pretrain_run_id moe_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0

echo "=== CLEAN SETTING: MoE-pretrained, Cora, lamda_env=0.1 ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/clean_setting_moe.py \
    --use_params --finetune_dataset cora --gpu 0 --debug --repeat 10 \
    --pretrain_run_id moe_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env 0.1

echo "=== CLEAN SETTING: MoE-pretrained, Pubmed, no lamda_env ==="
PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/clean_setting_moe.py \
    --use_params --finetune_dataset pubmed --gpu 0 --debug --repeat 10 \
    --pretrain_run_id moe_pretrain --pretrain_model_epoch 50 \
    --moe --moe_layers all --moe_experts 3 --moe_tau 1.0

echo "====== ALL DONE ======"
