#!/usr/bin/env bash
set -e

########################################
# CONFIG
########################################

MAX_JOBS=4
AVAILABLE_GPUS=(5)     # 你原来固定用 5；多卡就改成 (0 1 2 3 4 5 ...)
MAX_RETRIES=1
NUM_GPUS=${#AVAILABLE_GPUS[@]}

########################################
# SEMAPHORE
########################################

SEMAPHORE=/tmp/gs_semaphore_timefilter_pems04
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
  grep -Eq 'mse:[[:space:]]*[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?,[[:space:]]*mae:[[:space:]]*[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' "$log_file"
}

########################################
# SETTINGS
########################################

model_name=TimeFilter
seq_len=96

patchlens=(12 6 3)
betas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

mkdir -p logs
: > failures.txt

gpu_ptr=0

########################################
# MAIN LOOP
########################################

for pred_len in 12 24 48
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

      model_id="PEMS04_${seq_len}_${pred_len}_fcv_patch${loss_patchlen}_b${beta}"
      log_file="logs/${model_id}.log"

      if [ -f "$log_file" ] && is_finished "$log_file"; then
        echo "⏭ Skip: $model_id"
        echo >&9
        continue
      fi

      gpu_id=${AVAILABLE_GPUS[$gpu_ptr]}
      gpu_ptr=$(( (gpu_ptr + 1) % NUM_GPUS ))

      cmd="python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./data \
        --data_path PEMS04.npz \
        --model_id ${model_id} \
        --model ${model_name} \
        --data PEMS \
        --features M \
        --seq_len ${seq_len} \
        --pred_len ${pred_len} \
        --e_layers 2 \
        --enc_in 307 \
        --dec_in 307 \
        --c_out 307 \
        --patch_len 48 \
        --des Exp \
        --d_model 512 \
        --d_ff 1024 \
        --dropout 0.1 \
        --top_p 0.0 \
        --learning_rate 0.0005 \
        --batch_size 16 \
        --train_epochs 20 \
        --itr 1 \
        --use_norm 0 \
        --add_loss fcv \
        --loss_patchlen ${loss_patchlen} \
        --alpha_add_loss ${alpha_add} \
        --beta_add_loss ${beta}"

      run_job $gpu_id "$cmd" "$log_file" "$model_id" &

    done
  done
done

wait
echo "All TimeFilter PEMS04 fcv search jobs finished."