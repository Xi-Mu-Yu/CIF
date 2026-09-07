export CUDA_VISIBLE_DEVICES=0,1,2,3

CUDA_LAUNCH_BLOCKING=7
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/ADFTD/ \
  --model_id ADFTD-Dep_1_10_1_1 \
  --model TCN \
  --data ADFTD-Dependent \
  --e_layers 6 \
  --batch_size 128 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 1000 \
  --patience 10\
  --t 1 \
  --n 10 \
  --a 1 \
  --b 1\
  --gpu 0

  export CUDA_LAUNCH_BLOCKING=7

export CUDA_VISIBLE_DEVICES=4
 python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/APAVA/ \
  --model_id APAVA-Indep-HMBitCN \
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
  --a 1 \
  --b 1\
  --gpu 0


  export CUDA_VISIBLE_DEVICES=6
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/APAVA/ \
  --model_id APAVA-Indep-hmbitcn_no0126 \
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
  --b -0.6\
  --learnab True\
  --log_snr True \
  --snr_log_interval 10 \
  --gpu 0





  export CUDA_VISIBLE_DEVICES=3
 python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/APAVA/ \
  --model_id APAVA_Ablation/APAVA-Indep-_-1-13-1-1_learn \
  --model TCN \
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
  --t -1 \
  --n 13 \
  --a 1 \
  --b 1\
  --learnab True \
  --gpu 0


  # ceshicanshu

  
  export CUDA_VISIBLE_DEVICES=3
 python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/APAVA/ \
  --model_id APAVA_Ablation/APAVA-Indep-_1-9-1-2_learn_test \
  --model TCN \
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
  --a 1.0329\
  --b 1.0412\
  --learnab True \
  --gpu 0



  # gpu --6
CUDA_VISIBLE_DEVICES=6 python \
  -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ../Medformer_train/dataset/APAVA/ \
  --model_id APAVA-Indep_learnab_best \
  --model TCN \
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
  --a 1 \
  --b 1 \
  --learnab False \
  --gpu 0


  CUDA_VISIBLE_DEVICES=4 python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/ADFTD/ \
  --model_id ADFTD-Indep_1_10_-0.19_-0.27_learnab_Best_test \
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
  --b -0.27\
  --learnab True\
  --log_snr True \
  --snr_log_interval 10\
  --gpu 0


# test

CUDA_VISIBLE_DEVICES=6 python \
  -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ../Medformer_train/dataset/ADFTD/ \
  --model_id ADFTD-Indep_1_10_-0.19_-0.27_learnab_Best \
  --model TCN \
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
  --b -0.27\
  --learnab True\
  --gpu 0


# test


  CUDA_VISIBLE_DEVICES=6 python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/ADFTD/ \
  --model_id ADFTD-Indep_1_10_-0.19_-0.27_learnab_test \
  --model TCN \
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
  --a -0.3550\
  --b -0.2683 \
  --learnab True\
  --gpu 0



CUDA_VISIBLE_DEVICES=4 python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/PTB-XL/ \
  --model_id TCN/PTB-XL-Indep_1_12_-0.25_-0.25_ablearn \
  --model TCN \
  --data PTB-XL \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp_TCN_PTB_XL' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --t 1 \
  --n 12 \
  --a -0.25 \
  --b -0.25 \
  --learnab True\
  --gpu 0

# TBRAIN Dataset

CUDA_VISIBLE_DEVICES=4 python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/TDBRAIN/ \
  --model_id TDBRAIN-Indep_-1_25_1_1.1_nolearnab_best \
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
  --b 1.1\
  --log_snr True \
  --snr_log_interval 10\
  --learnab False\
  --gpu 0

  # test


  CUDA_VISIBLE_DEVICES=6 python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/TDBRAIN/ \
  --model_id TDBRAIN-Indep_-1_25_1_1.1_nolearnab_test \
  --model TCN \
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
  --a 1.0118 \
  --b 1.1176\
  --learnab False\
  --gpu 0

CUDA_VISIBLE_DEVICES=6 python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/PTB/ \
  --model_id PTB-Indep_1_8_0.21_-0.5_best_no6_7 \
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
  --b -0.5\
  --learnab False\
  --gpu 0

  # test

  CUDA_VISIBLE_DEVICES=6 python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/PTB/ \
  --model_id PTB-Indep_1_8_0.21_-0.5_test \
  --model TCN \
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
  --a 1.0105 \
  --b 1.0031\
  --learnab True\
  --gpu 0
# Two human activities dataset

# UCI-HAR Dataset
CUDA_VISIBLE_DEVICES=6
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/UCI-HAR/ \
  --model_id UCI-HAR \
  --model TCN \
  --data UCI-HAR \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --t 1 \
  --n 4 \
  --a 1 \
  --b 1\
  --learnab True\
  --gpu 0


noCIF

CUDA_VISIBLE_DEVICES=7
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/UCI-HAR/ \
  --model_id UCI-HAR_nocif \
  --model TCN \
  --data UCI-HAR \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10 \
  --t 1 \
  --n 4 \
  --a 1 \
  --b 1\
  --learnab True\
  --gpu 0

# FLAAP Dataset
  CUDA_VISIBLE_DEVICES=7 python \
    -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ../Medformer_train/dataset/FLAAP/ \
    --model_id FLAAP_HMbiTCN/FLAAP_1_3_-0.5_0.75_True \
    --model TCN \
    --data FLAAP \
    --e_layers 6 \
    --batch_size 32 \
    --d_model 128 \
    --d_ff 256 \
    --des 'Exp' \
    --itr 5 \
    --learning_rate 0.0001 \
    --train_epochs 100 \
    --patience 10\
    --t 1 \
    --n 3 \
    --a -0.5 \
    --b 0.75\
    --learnab True\
    --gpu 0



  CUDA_VISIBLE_DEVICES=6 python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/FLAAP/ \
  --model_id FLAAP_HMbiTCN/FLAAP_n=4_a0.5_b1_True \
  --model TCN \
  --data FLAAP \
  --e_layers 6 \
  --batch_size 32 \
  --d_model 128 \
  --d_ff 256 \
  --des 'Exp' \
  --itr 5 \
  --learning_rate 0.0001 \
  --train_epochs 100 \
  --patience 10\
  --t 1 \
  --n 4 \
  --a 0.5 \
  --b 1\
  --learnab True\
  --gpu 0