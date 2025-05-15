CUDA_VISIBLE_DEVICES=6 python \
  -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./dataset/APAVA/ \
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





  CUDA_VISIBLE_DEVICES=6 python \
  -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./dataset/ADFTD/ \
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





CUDA_VISIBLE_DEVICES=4 python \
  -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./dataset/PTB-XL/ \
  --model_id TCN/PTB-XL-Indep_1_12_-0.25_-0.25_ablearn_best \
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



  CUDA_VISIBLE_DEVICES=6 python \
  -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./dataset/TDBRAIN/ \
  --model_id TDBRAIN-Indep_-1_25_1_1.1_nolearnab_best \
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
  --a 1 \
  --b 1.1\
  --learnab False\
  --gpu 0



CUDA_VISIBLE_DEVICES=6 python \
  -u run.py \
  --task_name classification \
  --is_training 0 \
  --root_path ./dataset/PTB/ \
  --model_id PTB-Indep_1_8_0.21_-0.5_best \
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
  --a 0.21 \
  --b -0.5\
  --learnab False\
  --gpu 0