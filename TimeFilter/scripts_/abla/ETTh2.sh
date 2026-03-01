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

is_finished() {
  local log_file="$1"
  grep -Eq 'mse:[[:space:]]*[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?,[[:space:]]*mae:[[:space:]]*[0-9]+(\.[0-9]+)?([eE][-+]?[0-9]+)?' "$log_file"
}

model_name=TimeFilter
seq_len=96

pred_lens=(96 192 336 720)
patchlens=(6)
betas=(0 0.001 0.002 0.005 0.01 0.02 0.05 0.1 0.2 0.5 1.0)

mkdir -p logs
: > failures.txt

job_idx=0

submit_job() {
  local model_id=$1
  local cmd=$2
  local log_file="logs/${model_id}.log"

  read -u9

  if [ -f "$log_file" ] && is_finished "$log_file"; then
    echo "⏭ Skip (finished): $model_id"
    echo >&9
    return
  fi

  {
    gpu_id=$((job_idx % TOTAL_GPUS))
    job_idx=$((job_idx + 1))
    run_job $gpu_id "$cmd" "$log_file" "$model_id"
  } &
}

get_fixed_args_by_pred() {
  local pred_len=$1
  case $pred_len in
    96)
      echo "--e_layers 1 --d_layers 1 --factor 3 --patch_len 4 --dropout 0.8 --top_p 0.0 --d_model 128 --d_ff 256"
      ;;
    192)
      echo "--e_layers 1 --d_layers 1 --factor 3 --patch_len 4 --dropout 0.6 --d_model 128 --d_ff 256"
      ;;
    336)
      echo "--e_layers 2 --d_layers 1 --factor 3 --patch_len 8 --dropout 0.7 --top_p 0.0 --d_model 256 --d_ff 256"
      ;;
    720)
      echo "--e_layers 2 --d_layers 1 --factor 3 --patch_len 8 --dropout 0.3 --top_p 0.0 --d_model 256 --d_ff 256"
      ;;
    *)
      echo "Unsupported pred_len: $pred_len" >&2
      exit 1
      ;;
  esac
}

for pred_len in "${pred_lens[@]}"; do
  fixed_args="$(get_fixed_args_by_pred "$pred_len")"

  for patchlen in "${patchlens[@]}"; do
    for beta in "${betas[@]}"; do

      alpha=$(python - <<PY
b=float("${beta}")
a=1.0-b
print(f"{a:.6f}".rstrip('0').rstrip('.'))
PY
)

      model_id="ETTh2_${seq_len}_${pred_len}_fcv_lp${patchlen}_b${beta}"
      cmd="python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./data \
        --data_path ETTh2.csv \
        --model_id ${model_id} \
        --model ${model_name} \
        --data ETTh2 \
        --features M \
        --seq_len ${seq_len} \
        --label_len 48 \
        --pred_len ${pred_len} \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --des 'Exp' \
        --learning_rate 0.0001 \
        --batch_size 32 \
        --train_epochs 10 \
        --itr 1 \
        ${fixed_args} \
        --add_loss fcv \
        --loss_patchlen ${patchlen} \
        --alpha_add_loss ${alpha} \
        --beta_add_loss ${beta}"

      submit_job "$model_id" "$cmd"

    done
  done
done

wait
echo "All ETTh2 TimeFilter jobs finished."