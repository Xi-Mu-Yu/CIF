
# export CUDA_VISIBLE_DEVICES=6
#  python \
#   -u run.py \
#   --task_name classification \
#   --is_training 1 \
#   --root_path ./dataset/APAVA/ \
#   --model_id APAVA-Indep \
#   --model TCN \
#   --data APAVA \
#   --e_layers 6 \
#   --batch_size 32 \
#   --d_model 128 \
#   --d_ff 256 \
#   --des 'Exp_10' \
#   --itr 5 \
#   --learning_rate 0.0001 \
#   --train_epochs 100 \
#   --patience 10 \
#   --t 1 \
#   --n 9 \
#   --a 1 \
#   --b 1\
#   --gpu 0


#!/bin/bash

# export CUDA_VISIBLE_DEVICES=6

# t=1
# n=9

# # ���������������� a �� b ����������
# for a in 0.5 1 2 4; do
#   for b in 0.5 1 2 4; do
    
#     model_id="HMBiTCN/APAVA-Indep_A${a}_B${b}_T${t}_N${n}"
    
#     echo "Running experiment: ${model_id}"
    
#     python -u run.py \
#       --task_name classification \
#       --is_training 1 \
#       --root_path ../Medformer_train/dataset/APAVA/ \
#       --model_id ${model_id} \
#       --model HMBiTCN \
#       --data APAVA \
#       --e_layers 6 \
#       --batch_size 32 \
#       --d_model 128 \
#       --d_ff 256 \
#       --des 'Exp_10' \
#       --itr 5 \
#       --learning_rate 0.0001 \
#       --train_epochs 100 \
#       --patience 10 \
#       --t ${t} \
#       --n ${n} \
#       --a ${a} \
#       --b ${b} \
#       --learnab False\
#       --gpu 0

#   done
# done


# export CUDA_VISIBLE_DEVICES=6

# # t ������
# t_values=(1 -1)

# # a �� b ���������� -1 �� 1������ 0.25
# a_values=$(seq -1 0.1 1)
# b_values=$(seq -1 0.1 1)

# # n ����������
# n_values=$(seq 5 15)

# # learnab ������
# learnab_values=("False" "True")

# for t in "${t_values[@]}"; do
#   for a in $a_values; do
#     for b in $b_values; do
#       for n in $n_values; do
#         for learnab in "${learnab_values[@]}"; do
          
#           model_id="HMBiTCN/APAVA-Indep_A${a}_B${b}_T${t}_N${n}_L${learnab}"
          
#           echo "Running experiment: ${model_id}"
          
#           python -u run.py \
#             --task_name classification \
#             --is_training 1 \
#             --root_path ../Medformer_train/dataset/APAVA/ \
#             --model_id ${model_id} \
#             --model HMBiTCN \
#             --data APAVA \
#             --e_layers 6 \
#             --batch_size 32 \
#             --d_model 128 \
#             --d_ff 256 \
#             --des 'Exp_10' \
#             --itr 5 \
#             --learning_rate 0.0001 \
#             --train_epochs 100 \
#             --patience 10 \
#             --t ${t} \
#             --n ${n} \
#             --a ${a} \
#             --b ${b} \
#             --learnab ${learnab} \
#             --gpu 0

#         done
#       done
#     done
#   done
# done


export CUDA_VISIBLE_DEVICES=5

max_jobs=4  # ��������������������
job_count=0

t_values=(1 -1)
a_values=$(seq -1 0.2 1)
b_values=$(seq -1 0.2 1)
n_values=$(seq 7 12)
# learnab_values=("False" "True")
learnab_values=( "True" "False")

for t in "${t_values[@]}"; do
  for a in $a_values; do
    for b in $b_values; do
      for n in $n_values; do
        for learnab in "${learnab_values[@]}"; do
          
          model_id="HMBiTCN/ADFTD-new/APAVA-Indep_A${a}_B${b}_T${t}_N${n}_L${learnab}"
          
          echo "Running experiment: ${model_id}"
          
          python -u run.py \
            --task_name classification \
            --is_training 1 \
            --root_path ../Medformer_train/dataset/ADFTD/ \
            --model_id ${model_id} \
            --model HMBiTCN \
            --data ADFTD \
            --e_layers 6 \
            --batch_size 32 \
            --d_model 128 \
            --d_ff 256 \
            --des 'Exp_10' \
            --itr 5 \
            --learning_rate 0.0001 \
            --train_epochs 100 \
            --patience 10 \
            --t ${t} \
            --n ${n} \
            --a ${a} \
            --b ${b} \
            --learnab ${learnab} \
            --gpu 0 &  # ��������

          ((job_count++))
          if [ "$job_count" -ge "$max_jobs" ]; then
            wait  # ����������������
            job_count=0
          fi

        done
      done
    done
  done
done

wait  # ����������������������



# bash scripts/classification/TCN_ADFTD.sh
# sed -i 's/\r$//' scripts/classification/TCN_ADFTD.sh
