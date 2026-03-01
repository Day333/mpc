#!/bin/bash

WORK_DIR="./"

cd "$WORK_DIR" || { echo "Directory $WORK_DIR not found! Exiting."; exit 1; }

SCRIPTS=(
    "ECL.sh"
    "ETTh1.sh"
    "ETTh2.sh"
    "ETTm1.sh"
    "ETTm2.sh"
    "PEMS03.sh"
    "PEMS04.sh"
    "PEMS07.sh"
    "PEMS08.sh"
    "Solar.sh"
    "Traffic.sh"
    "Weather.sh"
)

echo "==============================================="
echo "Batch execution started at: $(date)"
echo "Working directory: $(pwd)"
echo "==============================================="

for script in "${SCRIPTS[@]}"; do
    SCRIPT_PATH="scripts/$script"
    
    echo ""
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Starting: $SCRIPT_PATH <<<"
    
    bash "$SCRIPT_PATH"
    
    if [ $? -ne 0 ]; then
        echo "!!! ERROR: $SCRIPT_PATH failed! Stopping the batch execution. !!!"
        exit 1
    fi
    
    echo ">>> [$(date '+%Y-%m-%d %H:%M:%S')] Finished: $SCRIPT_PATH <<<"
    echo "-----------------------------------------------"
done

echo ""
echo "==============================================="
echo "All scripts executed successfully at: $(date)"
echo "==============================================="