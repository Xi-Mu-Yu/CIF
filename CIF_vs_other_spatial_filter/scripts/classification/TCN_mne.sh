#!/usr/bin/env bash
# TCN classification: MNE spatial methods (ICA / SSP / SSS) on four datasets.
# Requires: pip install mne scipy

set -euo pipefail

GPU="${GPU:-6}"
MODES=(ica ssp sss linear)

for MODE in "${MODES[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU}" python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ../Medformer_train/dataset/APAVA/ \
    --model_id "APAVA-spatial_${MODE}" \
    --model TCN \
    --data APAVA \
    --e_layers 6 \
    --batch_size 32 \
    --d_model 128 \
    --d_ff 256 \
    --des "Exp_APAVA_spatial_${MODE}" \
    --itr 5 \
    --learning_rate 0.0001 \
    --train_epochs 100 \
    --patience 10 \
    --t 1 \
    --n 9 \
    --a -0.8 \
    --b -0.6 \
    --learnab True \
    --spatial_mode "${MODE}" \
    --gpu 0
done

for MODE in "${MODES[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU}" python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ../Medformer_train/dataset/ADFTD/ \
    --model_id "ADFTD-spatial_${MODE}" \
    --model TCN \
    --data ADFTD \
    --e_layers 6 \
    --batch_size 32 \
    --d_model 128 \
    --d_ff 256 \
    --des "Exp_ADFTD_spatial_${MODE}" \
    --itr 5 \
    --learning_rate 0.0001 \
    --train_epochs 100 \
    --patience 10 \
    --t 1 \
    --n 10 \
    --a -0.19 \
    --b -0.27 \
    --learnab True \
    --spatial_mode "${MODE}" \
    --gpu 0
done

for MODE in "${MODES[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU}" python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ../Medformer_train/dataset/TDBRAIN/ \
    --model_id "TDBRAIN-spatial_${MODE}" \
    --model TCN \
    --data TDBRAIN \
    --e_layers 6 \
    --batch_size 32 \
    --d_model 128 \
    --d_ff 256 \
    --des "Exp_TDBRAIN_spatial_${MODE}" \
    --itr 5 \
    --learning_rate 0.0001 \
    --train_epochs 100 \
    --patience 10 \
    --t -1 \
    --n 25 \
    --a 1 \
    --b 1.1 \
    --learnab False \
    --spatial_mode "${MODE}" \
    --gpu 0
done

for MODE in "${MODES[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPU}" python -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ../Medformer_train/dataset/PTB/ \
    --model_id "PTB-spatial_${MODE}" \
    --model TCN \
    --data PTB \
    --e_layers 6 \
    --batch_size 32 \
    --d_model 128 \
    --d_ff 256 \
    --des "Exp_PTB_spatial_${MODE}" \
    --itr 5 \
    --learning_rate 0.0001 \
    --train_epochs 100 \
    --patience 10 \
    --t 1 \
    --n 8 \
    --a 0.21 \
    --b -0.5 \
    --learnab False \
    --spatial_mode "${MODE}" \
    --gpu 0
done
