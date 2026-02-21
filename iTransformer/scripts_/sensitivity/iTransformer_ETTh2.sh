#!/usr/bin/env bash
set -e

MAX_JOBS=2
TOTAL_GPUS=2
MAX_RETRIES=1

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

# 判断log是否已包含最终指标（mse/mae）
is_finished() {
  local log_file="$1"
  # 支持：
  # mse:0.223..., mae:0.258...
  # mse: 0.223..., mae: 0.258...
  grep -Eq 'mse:[[:space:]]*[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?,[[:space:]]*mae:[[:space:]]*[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' "$log_file"
}

model_name=iTransformer

seq_len=96
pred_lens=(96 192 336 720)

patchlens=(2 4 8 16 32)
betas=(0 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0)

root_path=./dataset/ETT-small/
data_path=ETTh2.csv

job_idx=0
mkdir -p logs

for pred_len in "${pred_lens[@]}"; do
  for patchlen in "${patchlens[@]}"; do
    for beta in "${betas[@]}"; do

      read -u9  # semaphore token

      # alpha = 1 - beta
      alpha=$(python - <<PY
b=float("${beta}")
a=1.0-b
print(f"{a:.6f}".rstrip('0').rstrip('.'))
PY
)

      model_id="ETTh2_${seq_len}_${pred_len}_fcv_patch${patchlen}_b${beta}"
      log_file="logs/${model_id}.log"

      # ✅ 跳过已跑好的（log存在且含mse/mae行）
      if [ -f "$log_file" ] && is_finished "$log_file"; then
        echo "⏭ Skip (finished): $model_id"
        echo >&9   # 归还token，否则会占坑导致并行卡死
        continue
      fi

      {
        gpu_id=$((job_idx % TOTAL_GPUS))
        job_idx=$((job_idx + 1))

        cmd="python -u run.py \
          --is_training 1 \
          --root_path ${root_path} \
          --data_path ${data_path} \
          --model_id ${model_id} \
          --model ${model_name} \
          --data ETTh2 \
          --features M \
          --seq_len ${seq_len} \
          --pred_len ${pred_len} \
          --e_layers 2 \
          --enc_in 7 \
          --dec_in 7 \
          --c_out 7 \
          --des 'Exp' \
          --d_model 128 \
          --d_ff 128 \
          --itr 1 \
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
echo "All ETTh2 jobs finished."