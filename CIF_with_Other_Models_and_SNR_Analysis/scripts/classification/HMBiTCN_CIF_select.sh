#!/usr/bin/env bash
# HMBiTCN + CIF(select): apply CIF only on TRAIN-split channels with snr_delta_ch_db > 0
# (derived from Trail_snr/*_cif_trial_snr_gain_per_channel.csv).

export CUDA_VISIBLE_DEVICES=0

# APAVA: t=1, n=9, a=-0.8, b=-0.6  -> TRAIN: [3,4,5,7,8]
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/APAVA/ \
  --model_id APAVA-Indep-HMBiTCN-CIF-select \
  --model HMBiTCN \
  --data APAVA \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp_10' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --t 1 \
  --n 9 \
  --a -0.8 \
  --b -0.6 \
  --learnab True \
  --cif_select True \
  --cif_snr_gain_csv ./Trail_snr/APAVA_cif_trial_snr_gain_per_channel.csv \
  --cif_snr_splits TRAIN \
  --gpu 0

# ADFTD: t=1, n=10, a=-0.19, b=-0.27  -> TRAIN: [1,3,4,7,9]
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/ADFTD/ \
  --model_id ADFTD-Indep-HMBiTCN-CIF-select \
  --model HMBiTCN \
  --data ADFTD \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp_ADFTD' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --t 1 \
  --n 10 \
  --a -0.19 \
  --b -0.27 \
  --learnab True \
  --cif_select True \
  --cif_snr_gain_csv ./Trail_snr/ADFTD_cif_trial_snr_gain_per_channel.csv \
  --cif_snr_splits TRAIN \
  --gpu 0

# PTB: t=1, n=8, a=0.21, b=-0.5  -> TRAIN: [0,1,2,3,4,5,7]
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/PTB/ \
  --model_id PTB-Indep-HMBiTCN-CIF-select \
  --model HMBiTCN \
  --data PTB \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp_TCN' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --t 1 \
  --n 8 \
  --a 0.21 \
  --b -0.5 \
  --learnab False \
  --cif_select True \
  --cif_snr_gain_csv ./Trail_snr/PTB_cif_trial_snr_gain_per_channel.csv \
  --cif_snr_splits TRAIN \
  --gpu 0

# TDBRAIN: t=-1, n=25, a=1, b=1.1  -> TRAIN: [0,1,3,4,5,6,8,10,14,15,18,20,21,22]
python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/TDBRAIN/ \
  --model_id TDBRAIN-Indep-HMBiTCN-CIF-select \
  --model HMBiTCN \
  --data TDBRAIN \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --t -1 \
  --n 25 \
  --a 1 \
  --b 1.1 \
  --learnab False \
  --cif_select True \
  --cif_snr_gain_csv ./Trail_snr/TDBRAIN_cif_trial_snr_gain_per_channel.csv \
  --cif_snr_splits TRAIN \
  --gpu 0
