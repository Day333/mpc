#!/usr/bin/env bash
set -e

########################################
# CONFIG
########################################

MAX_JOBS=7
AVAILABLE_GPUS=(0 1 2 3 4 5 6 7)
MAX_RETRIES=1
NUM_GPUS=${#AVAILABLE_GPUS[@]}

########################################
# SEMAPHORE
########################################

SEMAPHORE=/tmp/gs_semaphore_tqnet_ettm2
mkfifo $SEMAPHORE
exec 9<>$SEMAPHORE
rm $SEMAPHORE

for ((i=0;i<${MAX_JOBS};i++)); do
    echo >&9
done

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

model_name=TQNet

root_path_name=./dataset/
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2

seq_len=96
enc_in=7
random_seed=2024

patchlens=(24 12 6 3)
betas=(0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0)

mkdir -p logs
: > failures.txt

gpu_ptr=0

########################################
# MAIN LOOP
########################################

for pred_len in 96 192 336 720
do
  for patchlen in "${patchlens[@]}"; do
    for beta in "${betas[@]}"; do

      read -u9

      alpha_add=$(python - <<PY
b=float("${beta}")
a=1.0-b
print(f"{a:.6f}".rstrip('0').rstrip('.'))
PY
)

      model_id="${model_id_name}_${seq_len}_${pred_len}_fcv_patch${patchlen}_b${beta}"
      log_file="logs/${model_id}.log"

      if [ -f "$log_file" ] && is_finished "$log_file"; then
          echo "⏭ Skip: $model_id"
          echo >&9
          continue
      fi

      gpu_id=${AVAILABLE_GPUS[$gpu_ptr]}
      gpu_ptr=$(( (gpu_ptr + 1) % NUM_GPUS ))

      cmd="python -u run.py \
        --is_training 1 \
        --root_path ${root_path_name} \
        --data_path ${data_path_name} \
        --model_id ${model_id} \
        --model ${model_name} \
        --data ${data_name} \
        --features M \
        --seq_len ${seq_len} \
        --pred_len ${pred_len} \
        --enc_in ${enc_in} \
        --cycle 96 \
        --train_epochs 30 \
        --patience 5 \
        --dropout 0.5 \
        --itr 1 \
        --batch_size 256 \
        --learning_rate 0.001 \
        --random_seed ${random_seed} \
        --add_loss fcv \
        --loss_patchlen ${patchlen} \
        --alpha_add_loss ${alpha_add} \
        --beta_add_loss ${beta}"

      run_job $gpu_id "$cmd" "$log_file" "$model_id" &

    done
  done
done

wait
echo "All TQNet ETTm2 fcv search jobs finished."