#!/bin/bash

WORK_DIR="./"

cd "$WORK_DIR" || { echo "Directory $WORK_DIR not found! Exiting."; exit 1; }

SCRIPT_DIR="scripts/multivariate_forecasting"

SCRIPTS=(
    "iTransformer_03.sh"
    "iTransformer_04.sh"
    "iTransformer_07.sh"
    "iTransformer_08.sh"
    "iTransformer_ECL.sh"
    "iTransformer_ETTh1.sh"
    "iTransformer_ETTh2.sh"
    "iTransformer_ETTm1.sh"
    "iTransformer_ETTm2.sh"
    "iTransformer_exchange.sh"
    "iTransformer_solar.sh"
    "iTransformer_traffic.sh"
    "iTransformer_weather.sh"
)

echo "=========================================================="
echo "iTransformer Batch Execution Started at: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================================="

for script in "${SCRIPTS[@]}"; do
    SCRIPT_PATH="$SCRIPT_DIR/$script"
    
    echo ""
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Starting: $SCRIPT_PATH <<<"
    
    bash "$SCRIPT_PATH"
    
    if [ $? -ne 0 ]; then
        echo "!!! ERROR: $SCRIPT_PATH failed! Stopping the batch execution. !!!"
        exit 1
    fi
    
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Finished: $SCRIPT_PATH <<<"
    echo "----------------------------------------------------------"
done

echo ""
echo "=========================================================="
echo "All iTransformer scripts executed successfully at: $(date)"
echo "=========================================================="