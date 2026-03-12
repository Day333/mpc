export CUDA_VISIBLE_DEVICES=2

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/CFPT" ]; then
    mkdir ./log/CFPT
fi

if [ ! -d "./log/CFPT/traffic" ]; then
    mkdir ./log/CFPT/traffic
fi

model_name=CFPT






for seq_len in 96
do
for pred_len in 192
do
for beta in 0.3
do
for seed in 2025
do
for learning_rate in 0.01
do
for batch_size in 4
do
for e_layers in 1
do
for d_model in 512
do
python -u run.py \
  --time_feature_types HourOfDay DayOfWeek \
  --task_name long_term_forecast \
  --is_training 1 \
  --with_curve 0 \
  --root_path ./dataset/traffic/ \
  --data_path traffic.csv \
  --model_id traffic_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --freq h \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 862 \
  --dec_in 862 \
  --c_out 862 \
  --des 'Exp' \
  --rda 4 \
  --rdb 1 \
  --ksize 5 \
  --beta $beta \
  --learning_rate $learning_rate \
  --batch_size $batch_size \
  --e_layers $e_layers \
  --d_model $d_model \
  --t_layers 3 \
  --patience 2 \
  --train_epochs 10 \
  --num_workers 10 \
  --dropout 0 \
  --loss mse \
  --period 24 \
  --seed $seed \
  --itr 1 | tee -a ./log/CFPT/traffic/$seq_len'_'$pred_len.txt
done
done
done
done
done
done
done
done
