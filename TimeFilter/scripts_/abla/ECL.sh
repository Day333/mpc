#!/usr/bin/env bash
set -e

############################################
# CONFIG
############################################

MAX_JOBS=2
AVAILABLE_GPUS=(0 1)
MAX_RETRIES=1
NUM_GPUS=${#AVAILABLE_GPUS[@]}

############################################
# SEMAPHORE
############################################

SEMAPHORE=/tmp/gs_semaphore_ecl
mkfifo $SEMAPHORE
exec 9<>$SEMAPHORE
rm $SEMAPHORE

for ((i=0;i<${MAX_JOBS};i++)); do
    echo >&9
done

############################################
# FUNCTIONS
############################################

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

############################################
# EXPERIMENT CONFIG
############################################

model_name=TimeFilter
mkdir -p logs
: > failures.txt

patchlens=(6)
betas=(0 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0)

gpu_ptr=0

########################################
# Short horizon
########################################

seq_len=96

for pred_len in 720 336 192 96
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

      model_id="ECL_${seq_len}_${pred_len}_fcv_patch${patchlen}_b${beta}"
      log_file="logs/${model_id}.log"

      if [ -f "$log_file" ] && is_finished "$log_file"; then
          echo "⏭ Skip (finished): $model_id"
          echo >&9
          continue
      fi

      gpu_id=${AVAILABLE_GPUS[$gpu_ptr]}
      gpu_ptr=$(( (gpu_ptr + 1) % NUM_GPUS ))

      if [[ "$pred_len" == "96" ]]; then
          dropout=0.5
      else
          dropout=0.4
      fi

      cmd="python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./data \
        --data_path electricity.csv \
        --model_id ${model_id} \
        --model ${model_name} \
        --data custom \
        --features M \
        --seq_len ${seq_len} \
        --label_len 48 \
        --pred_len ${pred_len} \
        --e_layers 2 \
        --d_layers 1 \
        --factor 3 \
        --enc_in 321 \
        --dec_in 321 \
        --c_out 321 \
        --patch_len 32 \
        --des Exp \
        --learning_rate 0.001 \
        --batch_size 16 \
        --train_epochs 15 \
        --d_model 512 \
        --d_ff 512 \
        --dropout ${dropout} \
        --itr 1 \
        --add_loss fcv \
        --loss_patchlen ${patchlen} \
        --alpha_add_loss ${alpha_add} \
        --beta_add_loss ${beta}"

      run_job $gpu_id "$cmd" "$log_file" "$model_id" &

    done
  done
done

# ########################################
# # Long horizon
# ########################################

# seq_len=512

# for pred_len in 96 192 336 720
# do
#   for patchlen in "${patchlens[@]}"; do
#     for beta in "${betas[@]}"; do

#       read -u9

#       alpha_add=$(python - <<PY
# b=float("${beta}")
# a=1.0-b
# print(f"{a:.6f}".rstrip('0').rstrip('.'))
# PY
# )

#       model_id="ECL_${seq_len}_${pred_len}_fcv_patch${patchlen}_b${beta}"
#       log_file="logs/${model_id}.log"

#       if [ -f "$log_file" ] && is_finished "$log_file"; then
#           echo "⏭ Skip (finished): $model_id"
#           echo >&9
#           continue
#       fi

#       gpu_id=${AVAILABLE_GPUS[$gpu_ptr]}
#       gpu_ptr=$(( (gpu_ptr + 1) % NUM_GPUS ))

#       cmd="python -u run.py \
#         --task_name long_term_forecast \
#         --is_training 1 \
#         --root_path ./data \
#         --data_path electricity.csv \
#         --model_id ${model_id} \
#         --model ${model_name} \
#         --data custom \
#         --features M \
#         --seq_len ${seq_len} \
#         --label_len 48 \
#         --pred_len ${pred_len} \
#         --e_layers 2 \
#         --d_layers 1 \
#         --factor 3 \
#         --enc_in 321 \
#         --dec_in 321 \
#         --c_out 321 \
#         --patch_len 128 \
#         --des Exp \
#         --learning_rate 0.001 \
#         --batch_size 16 \
#         --train_epochs 15 \
#         --d_model 512 \
#         --d_ff 512 \
#         --dropout 0.5 \
#         --top_p 0.0 \
#         --itr 1 \
#         --add_loss fcv \
#         --loss_patchlen ${patchlen} \
#         --alpha_add_loss ${alpha_add} \
#         --beta_add_loss ${beta}"

#       run_job $gpu_id "$cmd" "$log_file" "$model_id" &

#     done
#   done
# done

# wait
# echo "All TimeFilter ECL fcv search jobs finished."