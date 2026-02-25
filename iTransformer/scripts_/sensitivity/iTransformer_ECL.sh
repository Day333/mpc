#!/usr/bin/env bash
set -e

MAX_JOBS=1
AVAILABLE_GPUS=(0)
MAX_RETRIES=1

NUM_GPUS=${#AVAILABLE_GPUS[@]}

SEMAPHORE=/tmp/gs_semaphore
mkfifo $SEMAPHORE
exec 9<>$SEMAPHORE
rm $SEMAPHORE
for ((i=0;i<${MAX_JOBS};i++)); do echo >&9; done

run_job() {
  local gpu_id=$1
  local cmd=$2
  local log_file=$3
  local model_id=$4
  local attempt=0

  while (( attempt <= MAX_RETRIES )); do
    echo "▶ [GPU $gpu_id][Try $((attempt+1))] $model_id"
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd > "$log_file" 2>&1

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

model_name=iTransformer

seq_len=96
pred_lens=(96 192 336 720)

# patchlens=(2 4 8 16 24)
patchlens=(16)
betas=(0 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0)

root_path=./dataset/electricity/
data_path=electricity.csv

job_idx=0
mkdir -p logs
: > failures.txt

for pred_len in "${pred_lens[@]}"; do
  for patchlen in "${patchlens[@]}"; do
    for beta in "${betas[@]}"; do

      read -u9  

      alpha=$(python - <<PY
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

      # Calculate GPU ID outside the subshell to ensure proper cycling
      gpu_index=$((job_idx % NUM_GPUS))
      gpu_id=${AVAILABLE_GPUS[$gpu_index]}
      job_idx=$((job_idx + 1))

      {
        cmd="python -u run.py \
          --is_training 1 \
          --root_path ${root_path} \
          --data_path ${data_path} \
          --model_id ${model_id} \
          --model ${model_name} \
          --data custom \
          --features M \
          --seq_len ${seq_len} \
          --pred_len ${pred_len} \
          --e_layers 3 \
          --enc_in 321 \
          --dec_in 321 \
          --c_out 321 \
          --des 'Exp' \
          --d_model 512 \
          --d_ff 512 \
          --batch_size 16 \
          --learning_rate 0.0005 \
          --itr 1 \
          --add_loss scv \
          --loss_patchlen ${patchlen} \
          --alpha_add_loss ${alpha} \
          --beta_add_loss ${beta}"

        run_job $gpu_id "$cmd" "$log_file" "$model_id"
      } &

    done
  done
done

wait
echo "All ECL jobs finished."