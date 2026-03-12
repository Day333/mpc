export CUDA_VISIBLE_DEVICES=2

if [ ! -d "./log" ]; then
    mkdir ./log
fi

if [ ! -d "./log/CFPT" ]; then
    mkdir ./log/CFPT
fi

if [ ! -d "./log/CFPT/ecl" ]; then
    mkdir ./log/CFPT/ecl
fi

model_name=CFPT

seq_len=96

for pred_len in 96 192
do
for beta in 0.1
do
for seed in 2025
do
python -u run.py \
  --time_feature_types HourOfDay DayOfWeek \
  --task_name long_term_forecast \
  --is_training 1 \
  --with_curve 0 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --freq h \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --rda 8 \
  --rdb 1 \
  --ksize 2 \
  --beta $beta \
  --learning_rate 0.01 \
  --batch_size 16 \
  --train_epochs 10 \
  --num_workers 10 \
  --dropout 0.0 \
  --loss mse \
  --seed $seed \
  --itr 1 | tee -a ./log/CFPT/ecl/$seq_len'_'$pred_len.txt
done
done
done

for pred_len in 336 720
do
for beta in 0.1
do
for seed in 2025
do
python -u run.py \
  --time_feature_types HourOfDay DayOfWeek SeasonOfYear \
  --task_name long_term_forecast \
  --is_training 1 \
  --with_curve 0 \
  --root_path ./dataset/electricity/ \
  --data_path electricity.csv \
  --model_id ECL_$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --freq h \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --factor 3 \
  --enc_in 321 \
  --dec_in 321 \
  --c_out 321 \
  --des 'Exp' \
  --rda 8 \
  --rdb 1 \
  --ksize 2 \
  --beta $beta \
  --learning_rate 0.01 \
  --batch_size 16 \
  --train_epochs 10 \
  --num_workers 10 \
  --dropout 0.0 \
  --loss mse \
  --seed $seed \
  --itr 1 | tee -a ./log/CFPT/ecl/$seq_len'_'$pred_len.txt
done
done
done
