#!/bin/bash
set -e
source /home/lam23005/miniconda3/etc/profile.d/conda.sh
conda activate STEM-GNN
cd /home/lam23005/STEM-GNN
export PYTHONPATH=/home/lam23005/STEM-GNN/STEM-GNN

CKPT=/home/lam23005/STEM-GNN/ckpts/pretrain_model/default
EPOCH=50
EPOCHS=60
EARLY=30
REPEAT=10

run_config () {
  local name="$1"; local jac="$2"; local lip="$3"
  echo "=== CONFIG: $name (decoder_jac_coeff=$jac, encoder_lip_coeff=$lip) ==="
  python -u STEM-GNN/scripts/degree_shift_ood.py --use_params --finetune_dataset cora --gpu 0 --debug \
    --pretrain_path "$CKPT" --pretrain_model_epoch $EPOCH \
    --finetune_epochs $EPOCHS --early_stop $EARLY --repeat $REPEAT \
    --decoder_jac_coeff "$jac" --encoder_lip_coeff "$lip" 2>&1 | grep -A4 "^Summary over"
  echo
}

run_config "baseline_no_reg"           0.0   0.0
run_config "head_only_lip_reg"         1e-3  0.0
run_config "encoder_only_lip_reg"      0.0   1e-3
run_config "head_plus_encoder_lip_reg" 1e-3  1e-3

echo "DEGREE_OOD_ALL_DONE"
