export CUDA_VISIBLE_DEVICES=2

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/CFPT" ]; then
    mkdir ./log/CFPT
fi

if [ ! -d "./log/CFPT/etth1" ]; then
    mkdir ./log/CFPT/etth1
fi

model_name=CFPT


for seq_len in 96
do
for beta in 0.3
do
for seed in 2025
do
for batch_size in 16
do
for learning_rate in 0.0001
do
for e_layers in 3
do
for d_model in 512
do
python -u run.py \
  --time_feature_types HourOfDay \
  --task_name long_term_forecast \
  --is_training 1 \
  --with_curve 0 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --freq h \
  --seq_len $seq_len \
  --pred_len 720 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --rda 1 \
  --rdb 1 \
  --ksize 2 \
  --beta $beta \
  --learning_rate $learning_rate \
  --batch_size $batch_size \
  --e_layers $e_layers \
  --d_model $d_model \
  --t_layers 3 \
  --train_epochs 10 \
  --num_workers 10 \
  --dropout 0.0 \
  --loss mse \
  --seed $seed \
  --itr 1 | tee -a ./log/CFPT/etth1/$seq_len'_'720.txt
done
done
done
done
done
done
done

