#!/usr/bin/env bash
set -e

########################################
# CONFIG
########################################

MAX_JOBS=1
AVAILABLE_GPUS=(0)
MAX_RETRIES=1
NUM_GPUS=${#AVAILABLE_GPUS[@]}

########################################
# SEMAPHORE
########################################

SEMAPHORE=/tmp/gs_semaphore_pdf_electricity
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
    echo "▶ [GPU $gpu_id][Try $((attempt+1))] $model_id"
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd >> "$log_file" 2>&1

    if [ $? -eq 0 ]; then
      echo "✅ [GPU $gpu_id] Success: $model_id"
      break
    else
      echo "❌ [GPU $gpu_id] Failed: $model_id (Attempt $((attempt+1)))"
      attempt=$((attempt + 1))
      if (( attempt > MAX_RETRIES )); then
        echo "$cmd" >> failures.txt
      fi
    fi
  done

  echo >&9
}

is_finished() {
  local log_file="$1"
  grep -Eq 'mse:[[:space:]]*[0-9]+|mae:[[:space:]]*[0-9]+' "$log_file"
}

########################################
# SETTINGS
########################################

model_name=PDF
model_id_name=Electricity
data_name=custom

root_path_name=./dataset/
data_path_name=electricity.csv

seq_len=96
random_seed=2021

patchlens=(12)
betas=(0.6)

mkdir -p logs
: > failures.txt

gpu_ptr=0

########################################
# SEARCH
########################################

for pred_len in 96 192 336 720
do
  for loss_patchlen in "${patchlens[@]}"; do
    for beta in "${betas[@]}"; do

      read -u9

      alpha_add=$(python - <<PY
b=float("${beta}")
a=1.0-b
print(f"{a:.6f}".rstrip('0').rstrip('.'))
PY
)

      model_id="${model_id_name}_${seq_len}_${pred_len}_fcv_patch${loss_patchlen}_b${beta}"
      log_file="logs/${model_name}_${model_id}.log"

      if [ -f "$log_file" ] && is_finished "$log_file"; then
        echo "⏭ Skip: $model_id"
        echo >&9
        continue
      fi

      gpu_id=${AVAILABLE_GPUS[$gpu_ptr]}
      gpu_ptr=$(( (gpu_ptr + 1) % NUM_GPUS ))

      cmd="python -u run_longExp.py \
        --random_seed ${random_seed} \
        --is_training 1 \
        --root_path ${root_path_name} \
        --data_path ${data_path_name} \
        --model_id ${model_id_name}_${seq_len}_${pred_len} \
        --model ${model_name} \
        --data ${data_name} \
        --features M \
        --seq_len ${seq_len} \
        --pred_len ${pred_len} \
        --enc_in 321 \
        --e_layers 3 \
        --n_heads 32 \
        --d_model 128 \
        --d_ff 256 \
        --dropout 0.45 \
        --fc_dropout 0.15 \
        --kernel_list 3 3 5 \
        --period 8 12 24 168 336 \
        --patch_len 1 2 3 16 24 \
        --stride 1 2 3 16 24 \
        --des Exp \
        --train_epochs 100 \
        --lradj TST \
        --pct_start 0.2 \
        --patience 10 \
        --itr 1 \
        --batch_size 32 \
        --learning_rate 0.0001 \
        --add_loss fcv \
        --loss_patchlen ${loss_patchlen} \
        --alpha_add_loss ${alpha_add} \
        --beta_add_loss ${beta}"

      run_job $gpu_id "$cmd" "$log_file" "$model_id" &

    done
  done
done

wait
echo "All PDF Electricity FCV search jobs finished."