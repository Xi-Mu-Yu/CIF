export CUDA_VISIBLE_DEVICES=6
 python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/APAVA/ \
  --model_id APAVA-Indep-CIF \
  --model HMBiTCNbac \
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




  
  CUDA_VISIBLE_DEVICES=6 
  python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/ADFTD/ \
  --model_id ADFTD-Indep-CIF \
  --model HMBiTCNbac \
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

  
CUDA_VISIBLE_DEVICES=6 
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/TDBRAIN/ \
  --model_id TDBRAIN-Indep-CIF \
  --model HMBiTCNbac \
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
  --learnab False\
  --gpu 0


  CUDA_VISIBLE_DEVICES=6 
  python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/PTB/ \
  --model_id PTB-Indep-CIF \
  --model HMBiTCNbac \
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


  
export CUDA_VISIBLE_DEVICES=3
  python \
    -u run.py \
    --task_name classification \
    --is_training 1 \
    --root_path ../Medformer_train/dataset/FLAAP/ \
    --model_id FLAAP_HMbiTCNbac-CIF \
    --model HMBiTCNbac \
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


    
export CUDA_VISIBLE_DEVICES=3
python \
  -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ../Medformer_train/dataset/UCI-HAR/ \
  --model_id UCI-HAR_nocif \
  --model HMBiTCNbac \
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