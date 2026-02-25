#!/usr/bin/env bash
set -e

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

############################################
# Parallel config
############################################
MAX_JOBS=6
AVAILABLE_GPUS=(1 2 3 4 5 6)
MAX_RETRIES=1

NUM_GPUS=${#AVAILABLE_GPUS[@]}

SEMAPHORE=/tmp/gs_semaphore_electricity_tqnet
mkfifo $SEMAPHORE
exec 9<>$SEMAPHORE
rm $SEMAPHORE
for ((i=0;i<${MAX_JOBS};i++)); do echo >&9; done

############################################
# Functions
############################################

run_job() {
  local gpu_id=$1
  local cmd=$2
  local log_file=$3
  local model_id=$4
  local attempt=0

  while (( attempt <= MAX_RETRIES )); do
    echo "▶ [GPU $gpu_id][Try $((attempt+1))] $model_id"

    echo "===== Attempt $((attempt+1)) =====" >> "$log_file"
    CUDA_VISIBLE_DEVICES=$gpu_id $cmd >> "$log_file" 2>&1

    if [ $? -eq 0 ]; then
      echo "✅ [GPU $gpu_id] Success: $model_id"
      break
    else
      echo "❌ [GPU $gpu_id] Failed: $model_id (Attempt $((attempt+1)))"
      attempt=$((attempt + 1))
      if (( attempt > MAX_RETRIES )); then
        echo "$cmd" >> failures_electricity_tqnet.txt
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
# Experiment config
############################################

model_name=TQNet
root_path_name=./dataset/
data_path_name=electricity.csv
model_id_name=Electricity
data_name=custom

seq_len=96
enc_in=321
random_seed=2024

# pred_lens=(96 192 336 720)
pred_lens=(192 336 720)

patchlens=(24)
betas=(0 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0)

mkdir -p logs
: > failures_electricity_tqnet.txt

job_idx=0

############################################
# Main loop
############################################

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

      model_id="${model_id_name}_${seq_len}_${pred_len}_fcv_patch${patchlen}_b${beta}"
      log_file="logs/${model_id}.log"

      if [ -f "$log_file" ] && is_finished "$log_file"; then
        echo "⏭ Skip (finished): $model_id"
        echo >&9
        continue
      fi

      gpu_index=$((job_idx % NUM_GPUS))
      gpu_id=${AVAILABLE_GPUS[$gpu_index]}
      job_idx=$((job_idx + 1))

      {
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
          --cycle 168 \
          --train_epochs 30 \
          --patience 5 \
          --itr 1 \
          --batch_size 32 \
          --learning_rate 0.003 \
          --random_seed ${random_seed} \
          --add_loss fcv \
          --loss_patchlen ${patchlen} \
          --alpha_add_loss ${alpha} \
          --beta_add_loss ${beta}"

        run_job $gpu_id "$cmd" "$log_file" "$model_id"

      } &

    done
  done
done

wait
echo "All electricity TQNet add_loss jobs finished."