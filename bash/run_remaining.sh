#!/bin/bash
#SBATCH --job-name=stem-gnn-cont
#SBATCH --partition=general-gpu
#SBATCH --account=jhf24001
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/lam23005/STEM-GNN/logs/stem_gnn_cont_%j.out
#SBATCH --error=/home/lam23005/STEM-GNN/logs/stem_gnn_cont_%j.err

source ~/miniconda3/bin/activate STEM-GNN
cd /home/lam23005/STEM-GNN

echo "====== FINETUNE: FB15K237 ======"
python STEM-GNN/finetune.py --use_params --finetune_dataset FB15K237 --gpu 0 --debug

echo "====== FINETUNE: chemhiv ======"
python STEM-GNN/finetune.py --use_params --finetune_dataset chemhiv --gpu 0 --debug

echo "====== FINETUNE: chempcba ======"
python STEM-GNN/finetune.py --use_params --finetune_dataset chempcba --gpu 0 --debug

echo "====== SCRIPT: degree_shift_ood ======"
python STEM-GNN/scripts/degree_shift_ood.py --use_params --finetune_dataset cora --gpu 0 --debug

echo "====== SCRIPT: homophily_shift_ood ======"
python STEM-GNN/scripts/homophily_shift_ood.py --use_params --finetune_dataset cora --gpu 0 --debug

echo "====== SCRIPT: missing_feature ======"
python STEM-GNN/scripts/missing_feature.py --use_params --finetune_dataset cora --gpu 0 --debug

echo "====== SCRIPT: random_edge_drop ======"
python STEM-GNN/scripts/random_edge_drop.py --use_params --finetune_dataset cora --gpu 0 --debug

echo "====== SCRIPT: tri_objective ======"
python STEM-GNN/scripts/tri_objective.py --use_params --finetune_dataset cora --gpu 0 --debug

echo "====== ALL DONE ======"
