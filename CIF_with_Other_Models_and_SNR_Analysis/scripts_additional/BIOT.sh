#!/bin/bash

export CUDA_VISIBLE_DEVICES=4

# APAVA - with CIF
# python \
#   -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ../Medformer_train/dataset/APAVA/ \
#   --model_id APAVA-Indep-BIOT-CIF \
#   --model BIOT \
#   --data APAVA \
#   --e_layers 6 \
#   --batch_size 32 \
#   --d_model 128 \
#   --d_ff 256 \
#   --patch_len 16 \
#   --des 'Exp_BIOT' \
#   --itr 5 \
#   --learning_rate 0.0001 \
#   --train_epochs 100 \
#   --patience 10 \
#   --use_cif True \
#   --t 1 \
#   --n 9 \
#   --a -0.8 \
#   --b -0.6 \
#   --learnab True \
#   --gpu 0


# APAVA - without CIF
# python \
#   -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ../Medformer_train/dataset/APAVA/ \
#   --model_id APAVA-Indep-BIOT-noCIF \
#   --model BIOT \
#   --data APAVA \
#   --e_layers 6 \
#   --batch_size 32 \
#   --d_model 128 \
#   --d_ff 256 \
#   --patch_len 16 \
#   --des 'Exp_BIOT' \
#   --itr 5 \
#   --learning_rate 0.0001 \
#   --train_epochs 100 \
#   --patience 10 \
#   --use_cif False \
#   --gpu 0


# ADFTD - with CIF
# python \
#   -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ../Medformer_train/dataset/ADFTD/ \
#   --model_id ADFTD-Indep-BIOT-CIF \
#   --model BIOT \
#   --data ADFTD \
#   --e_layers 6 \
#   --batch_size 32 \
#   --d_model 128 \
#   --d_ff 256 \
#   --patch_len 16 \
#   --des 'Exp_BIOT' \
#   --itr 5 \
#   --learning_rate 0.0001 \
#   --train_epochs 100 \
#   --patience 10 \
#   --use_cif True \
#   --t 1 \
#   --n 10 \
#   --a -0.19 \
#   --b -0.27 \
#   --learnab True \
#   --gpu 0


# ADFTD - without CIF
# python \
#   -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ../Medformer_train/dataset/ADFTD/ \
#   --model_id ADFTD-Indep-BIOT-noCIF \
#   --model BIOT \
#   --data ADFTD \
#   --e_layers 6 \
#   --batch_size 32 \
#   --d_model 128 \
#   --d_ff 256 \
#   --patch_len 16 \
#   --des 'Exp_BIOT' \
#   --itr 5 \
#   --learning_rate 0.0001 \
#   --train_epochs 100 \
#   --patience 10 \
#   --use_cif False \
#   --gpu 0


# TDBRAIN - with CIF
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/TDBRAIN/ \
  --model_id TDBRAIN-Indep-BIOT-CIF \
  --model BIOT \
  --data TDBRAIN \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --patch_len 16 \
  --des 'Exp_BIOT' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --use_cif True \
  --t -1 \
  --n 25 \
  --a 1 \
  --b 1.1 \
  --learnab False \
  --gpu 0


# TDBRAIN - without CIF
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/TDBRAIN/ \
  --model_id TDBRAIN-Indep-BIOT-noCIF \
  --model BIOT \
  --data TDBRAIN \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --patch_len 16 \
  --des 'Exp_BIOT' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --use_cif False \
  --gpu 0


# PTB - with CIF
# python \
#   -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ../Medformer_train/dataset/PTB/ \
#   --model_id PTB-Indep-BIOT-CIF \
#   --model BIOT \
#   --data PTB \
#   --e_layers 6 \
#   --batch_size 32 \
#   --d_model 128 \
#   --d_ff 256 \
#   --patch_len 16 \
#   --des 'Exp_BIOT' \
#   --itr 5 \
#   --learning_rate 0.0001 \
#   --train_epochs 100 \
#   --patience 10 \
#   --use_cif True \
#   --t 1 \
#   --n 8 \
#   --a 0.21 \
#   --b -0.5 \
#   --learnab False \
#   --gpu 0


# PTB - without CIF
# python \
#   -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ../Medformer_train/dataset/PTB/ \
#   --model_id PTB-Indep-BIOT-noCIF \
#   --model BIOT \
#   --data PTB \
#   --e_layers 6 \
#   --batch_size 32 \
#   --d_model 128 \
#   --d_ff 256 \
#   --patch_len 16 \
#   --des 'Exp_BIOT' \
#   --itr 5 \
#   --learning_rate 0.0001 \
#   --train_epochs 100 \
#   --patience 10 \
#   --use_cif False \
#   --gpu 0
