# Ablation Studies
# Modify the boolean values of parameters `self.use_tq` and `self.channel_aggre` in `models/TQNet` to perform ablation studies
#1. When `self.use_tq=True` and `self.channel_aggre=True`, it represents the original TQNet.
#2. When `self.use_tq=True` and `self.channel_aggre=False`, it indicates the removal of the attention module, but the TQ design is retained, and this part becomes the channel identifier module.
#3. When `self.use_tq=False` and `self.channel_aggre=True`, it indicates the removal of TQ, but the attention module is retained, and this part becomes the self-attention module.
#4. When `self.use_tq=False` and `self.channel_aggre=False`, it represents the removal of both TQ and the attention module, leaving only a basic MLP module.


model_name=TQNet

root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

seq_len=96
for pred_len in 96 192 336 720
do
for random_seed in 2024
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 321 \
      --cycle 168 \
      --train_epochs 30 \
      --patience 5 \
      --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed
done
done


root_path_name=./dataset/
data_path_name=PEMS03.npz
model_id_name=PEMS03
data_name=PEMS

seq_len=96
for pred_len in 12 24 48 96
do
for random_seed in 2024
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 358 \
      --cycle 288 \
      --train_epochs 30 \
      --patience 5 \
      --use_revin 0 \
      --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed
done
done


root_path_name=./dataset/
data_path_name=PEMS04.npz
model_id_name=PEMS04
data_name=PEMS

seq_len=96
for pred_len in 12 24 48 96
do
for random_seed in 2024
do
    python -u run.py \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id $model_id_name'_'$seq_len'_'$pred_len \
      --model $model_name \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --pred_len $pred_len \
      --enc_in 307 \
      --cycle 288 \
      --train_epochs 30 \
      --patience 5 \
      --use_revin 0 \
      --itr 1 --batch_size 32 --learning_rate 0.003 --random_seed $random_seed
done
done

