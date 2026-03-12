export CUDA_VISIBLE_DEVICES=2

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/CFPT" ]; then
    mkdir ./log/CFPT
fi

if [ ! -d "./log/CFPT/weather" ]; then
    mkdir ./log/CFPT/weather
fi

model_name=CFPT

for seq_len in 96
do
for pred_len in 96
do
for beta in 0.6
do
for seed in 2025
do
python -u run.py \
  --time_feature_types HourOfDay SeasonOfYear \
  --task_name long_term_forecast \
  --is_training 1 \
  --with_curve 0 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --freq t \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --beta $beta \
  --learning_rate 0.005 \
  --e_layers 3 \
  --d_model 512 \
  --batch_size 128 \
  --t_layers 3 \
  --train_epochs 10 \
  --num_workers 10 \
  --dropout 0 \
  --loss mse \
  --seed $seed \
  --itr 1 | tee -a ./log/CFPT/weather/$seq_len'_'$pred_len.txt
done
done
done
done
