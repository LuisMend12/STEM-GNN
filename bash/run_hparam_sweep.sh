#!/bin/bash
#SBATCH --job-name=stem-hparam-sweep
#SBATCH --partition=general-gpu
#SBATCH --account=jhf24001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lam23005/STEM-GNN/logs/hparam_sweep_%j.out
#SBATCH --error=/home/lam23005/STEM-GNN/logs/hparam_sweep_%j.err

source ~/miniconda3/bin/activate STEM-GNN
cd /home/lam23005/STEM-GNN

# lamda_env sweep: how much entropy regularization is needed?
for LAM in 0.0 0.05 0.1 0.2 0.5; do
    echo "=== lamda_env=$LAM (Cora, MoE) ==="
    PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
        --use_params --finetune_dataset cora --gpu 0 --debug --repeat 5 \
        --pretrain_dataset na \
        --moe --moe_layers all --moe_experts 3 --moe_tau 1.0 --lamda_env $LAM
done

# num_experts sweep: does more experts help?
for K in 2 3 5; do
    echo "=== moe_experts=$K (Cora, lamda_env=0.1) ==="
    PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
        --use_params --finetune_dataset cora --gpu 0 --debug --repeat 5 \
        --pretrain_dataset na \
        --moe --moe_layers all --moe_experts $K --moe_tau 1.0 --lamda_env 0.1
done

# tau sweep: routing sharpness
for TAU in 0.5 1.0 2.0; do
    echo "=== moe_tau=$TAU (Cora, 3 experts, lamda_env=0.1) ==="
    PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN python STEM-GNN/scripts/degree_shift_ood.py \
        --use_params --finetune_dataset cora --gpu 0 --debug --repeat 5 \
        --pretrain_dataset na \
        --moe --moe_layers all --moe_experts 3 --moe_tau $TAU --lamda_env 0.1
done

echo "====== ALL DONE ======"
