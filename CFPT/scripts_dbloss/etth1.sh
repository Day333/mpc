#!/usr/bin/env bash
set -e

########################################
# CONFIG
########################################

MAX_JOBS=4
AVAILABLE_GPUS=(0 2 3 6)
MAX_RETRIES=1
NUM_GPUS=${#AVAILABLE_GPUS[@]}

########################################
# SEMAPHORE
########################################

SEMAPHORE=/tmp/gs_semaphore_cfpt_etth1
mkfifo $SEMAPHORE
exec 9<>$SEMAPHORE
rm $SEMAPHORE

for ((i=0;i<${MAX_JOBS};i++)); do echo >&9; done

########################################
# FUNCTIONS
########################################

run_job() {
  local gpu_id=$1
  local cmd=$2
  local log_file=$3
  local model_id=$4
  local attempt=0

  while (( attempt <= MAX_RETRIES )); do
    echo "▶ [GPU $gpu_id] $model_id"
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd >> "$log_file" 2>&1

    if [ $? -eq 0 ]; then
      echo "✅ Success: $model_id"
      break
    else
      echo "❌ Failed: $model_id"
      attempt=$((attempt + 1))
    fi
  done

  echo >&9
}

is_finished() {
  local log_file="$1"
  grep -Eq 'mse:[[:space:]]*[0-9]+' "$log_file"
}

########################################
# SETTINGS
########################################

model_name=CFPT
seq_len=96
seed=2025
d_model=512

mkdir -p logs

gpu_ptr=0

########################################
# LOOP
########################################

for pred_len in 96 192 336 720
do

  if [[ "$pred_len" == "96" ]]; then
      beta_model=0.5
      batch_size=8
      ksize=3
  elif [[ "$pred_len" == "192" ]]; then
      beta_model=0.5
      batch_size=16
      ksize=3
  elif [[ "$pred_len" == "336" ]]; then
      beta_model=0.7
      batch_size=16
      ksize=5
  else
      beta_model=0.3
      batch_size=16
      ksize=2
  fi

  read -u9

  model_id="ETTh1_${seq_len}_${pred_len}_base"
  log_file="logs/${model_name}_${model_id}.log"

  if [ -f "$log_file" ] && is_finished "$log_file"; then
      echo "⏭ Skip: $model_id"
      echo >&9
      continue
  fi

  gpu_id=${AVAILABLE_GPUS[$gpu_ptr]}
  gpu_ptr=$(( (gpu_ptr + 1) % NUM_GPUS ))

  cmd="python -u run.py \
      --time_feature_types HourOfDay \
      --task_name long_term_forecast \
      --is_training 1 \
      --with_curve 0 \
      --root_path ./dataset/ETT-small/ \
      --data_path ETTh1.csv \
      --model_id ETTh1_${seq_len}_${pred_len} \
      --model CFPT \
      --data ETTh1 \
      --features M \
      --freq h \
      --seq_len ${seq_len} \
      --pred_len ${pred_len} \
      --factor 3 \
      --enc_in 7 \
      --dec_in 7 \
      --c_out 7 \
      --des Exp \
      --rda 1 \
      --rdb 1 \
      --ksize ${ksize} \
      --beta ${beta_model} \
      --learning_rate 0.0001 \
      --batch_size ${batch_size} \
      --e_layers 3 \
      --d_model ${d_model} \
      --t_layers 3 \
      --train_epochs 10 \
      --num_workers 10 \
      --dropout 0.0 \
      --loss DBLoss \
      --seed ${seed} \
      --itr 1"

  run_job $gpu_id "$cmd" "$log_file" "$model_id" &

done

wait
echo "All CFPT ETTh1 base jobs finished."