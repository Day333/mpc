#!/usr/bin/env bash
set -e

########################################
# CONFIG
########################################

MAX_JOBS=6
AVAILABLE_GPUS=(0 1 2 3 4 5 6) 
MAX_RETRIES=1
NUM_GPUS=${#AVAILABLE_GPUS[@]}

########################################
# SEMAPHORE
########################################

SEMAPHORE=/tmp/gs_semaphore_timefilter_ettm1
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
# EXPERIMENT SETTINGS
########################################

model_name=TimeFilter
seq_len=96

patchlens=(24 12 6 3)
betas=(0 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0)

mkdir -p logs
: > failures.txt

gpu_ptr=0

########################################
# MAIN LOOP
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

      # 保持你原始的不同 horizon 配置
      if [[ "$pred_len" == "96" ]]; then
          dropout=0.3
          base_patch=8
      elif [[ "$pred_len" == "192" || "$pred_len" == "336" ]]; then
          dropout=0.5
          base_patch=8
      else
          dropout=0.7
          base_patch=16
      fi

      model_id="ETTm1_${seq_len}_${pred_len}_fcv_patch${loss_patchlen}_b${beta}"
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
        --data_path ETTm1.csv \
        --model_id ${model_id} \
        --model ${model_name} \
        --data ETTm1 \
        --features M \
        --seq_len ${seq_len} \
        --label_len 48 \
        --pred_len ${pred_len} \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --dropout ${dropout} \
        --top_p 0.0 \
        --patch_len ${base_patch} \
        --des Exp \
        --learning_rate 0.0001 \
        --batch_size 32 \
        --train_epochs 10 \
        --d_model 256 \
        --d_ff 256 \
        --itr 1 \
        --add_loss fcv \
        --loss_patchlen ${loss_patchlen} \
        --alpha_add_loss ${alpha_add} \
        --beta_add_loss ${beta}"

      run_job $gpu_id "$cmd" "$log_file" "$model_id" &

    done
  done
done

wait
echo "All TimeFilter ETTm1 fcv search jobs finished."