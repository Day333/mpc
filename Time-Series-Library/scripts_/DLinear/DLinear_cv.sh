#!/usr/bin/env bash
set -e

########################################
# GPU & Concurrency Config
########################################

MAX_JOBS=2                 # 同时运行的最大任务数
AVAILABLE_GPUS=(5 6)   # 可用的 GPU 列表
MAX_RETRIES=1              # 失败重试次数
NUM_GPUS=${#AVAILABLE_GPUS[@]}

mkdir -p logs
gpu_ptr=0

########################################
# Basic config
########################################

MODEL_NAME=DLinear          # change to PatchTST if needed
SEQ_LEN=96
PRED_LENS=(96 192 336 720)

# ======================================
# Add Loss Config
# ======================================
LOSS_PATCHLEN=6
ALPHA_ADD_LOSS=0.3
BETA_ADD_LOSS=0.7

ROOT_ETT=./dataset/ETT-small/
ROOT_WEATHER=./dataset/weather/
ROOT_ECL=./dataset/electricity/
ROOT_TRAFFIC=./dataset/traffic/

########################################
# Semaphore Setup
########################################

SEMAPHORE=/tmp/gs_semaphore_benchmark_$$
mkfifo $SEMAPHORE
exec 9<>$SEMAPHORE
rm $SEMAPHORE

for ((i=0;i<${MAX_JOBS};i++)); do
  echo >&9
done

########################################
# Job Runner
########################################

run_job() {
  local gpu_id=$1
  local log_file=$2
  local model_id=$3
  local cmd=$4
  local attempt=0

  while (( attempt <= MAX_RETRIES )); do
    echo "▶ [GPU $gpu_id][Try $((attempt+1))] $model_id"
    
    if eval "CUDA_VISIBLE_DEVICES=$gpu_id $cmd > \"$log_file\" 2>&1"; then
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

########################################
# Shared runner
########################################

run_exp() {
  local root_path=$1
  local data_path=$2
  local model_id=$3
  local data_name=$4
  local enc_in=$5
  local pred_len=$6
  local extra_args=$7

  read -u9

  local gpu_id=${AVAILABLE_GPUS[$gpu_ptr]}
  gpu_ptr=$(( (gpu_ptr + 1) % NUM_GPUS ))

  local log_file="logs/${MODEL_NAME}_${model_id}.log"
  local cmd="python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path \"${root_path}\" \
    --data_path \"${data_path}\" \
    --model_id \"${model_id}\" \
    --model \"${MODEL_NAME}\" \
    --data \"${data_name}\" \
    --features M \
    --seq_len ${SEQ_LEN} \
    --label_len 48 \
    --pred_len ${pred_len} \
    --e_layers 2 \
    --d_layers 1 \
    --factor 3 \
    --enc_in ${enc_in} \
    --dec_in ${enc_in} \
    --c_out ${enc_in} \
    --des Exp \
    --itr 1 \
    --add_loss fcv \
    --loss_patchlen ${LOSS_PATCHLEN} \
    --alpha_add_loss ${ALPHA_ADD_LOSS} \
    --beta_add_loss ${BETA_ADD_LOSS} \
    ${extra_args}"

  run_job "$gpu_id" "$log_file" "$model_id" "$cmd" &
}

########################################
# Model-specific args
########################################

get_model_args() {
  local dataset_tag=$1
  local pred_len=$2

  if [[ "${MODEL_NAME}" == "DLinear" ]]; then
    echo ""
    return
  fi

  if [[ "${MODEL_NAME}" == "PatchTST" ]]; then
    echo "--fc_dropout 0.2 \
          --head_dropout 0.0 \
          --patch_len 16 \
          --stride 8 \
          --d_model 128 \
          --d_ff 256 \
          --n_heads 4 \
          --dropout 0.2"
    return
  fi

  echo ""
}

########################################
# Benchmark Loops
########################################

# ETTh1
for pred_len in "${PRED_LENS[@]}"; do
  extra_args=$(get_model_args ETTh1 ${pred_len})
  run_exp "${ROOT_ETT}" "ETTh1.csv" "ETTh1_${SEQ_LEN}_${pred_len}" "ETTh1" 7 ${pred_len} "${extra_args}"
done

# ETTh2
for pred_len in "${PRED_LENS[@]}"; do
  extra_args=$(get_model_args ETTh2 ${pred_len})
  run_exp "${ROOT_ETT}" "ETTh2.csv" "ETTh2_${SEQ_LEN}_${pred_len}" "ETTh2" 7 ${pred_len} "${extra_args}"
done

# ETTm1
for pred_len in "${PRED_LENS[@]}"; do
  extra_args=$(get_model_args ETTm1 ${pred_len})
  run_exp "${ROOT_ETT}" "ETTm1.csv" "ETTm1_${SEQ_LEN}_${pred_len}" "ETTm1" 7 ${pred_len} "${extra_args}"
done

# ETTm2
for pred_len in "${PRED_LENS[@]}"; do
  extra_args=$(get_model_args ETTm2 ${pred_len})
  run_exp "${ROOT_ETT}" "ETTm2.csv" "ETTm2_${SEQ_LEN}_${pred_len}" "ETTm2" 7 ${pred_len} "${extra_args}"
done

# Weather
for pred_len in "${PRED_LENS[@]}"; do
  extra_args=$(get_model_args Weather ${pred_len})
  run_exp "${ROOT_WEATHER}" "weather.csv" "weather_${SEQ_LEN}_${pred_len}" "custom" 21 ${pred_len} "${extra_args}"
done

# ECL
for pred_len in "${PRED_LENS[@]}"; do
  extra_args=$(get_model_args ECL ${pred_len})
  run_exp "${ROOT_ECL}" "electricity.csv" "ECL_${SEQ_LEN}_${pred_len}" "custom" 321 ${pred_len} "${extra_args}"
done

# Traffic
for pred_len in "${PRED_LENS[@]}"; do
  extra_args=$(get_model_args Traffic ${pred_len})
  run_exp "${ROOT_TRAFFIC}" "traffic.csv" "traffic_${SEQ_LEN}_${pred_len}" "custom" 862 ${pred_len} "${extra_args}"
done

########################################
# Wait for completion
########################################

wait
echo "🎉 All benchmark jobs finished."