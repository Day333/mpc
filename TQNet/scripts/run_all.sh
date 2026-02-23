#!/bin/bash

# Uncomment the following line if you want the execution to stop immediately if any individual script fails
# set -e

# Define the directory containing the scripts
SCRIPT_DIR="scripts/TQNet"

# Check if the directory exists
if [ ! -d "$SCRIPT_DIR" ]; then
    echo "Error: Directory $SCRIPT_DIR not found."
    exit 1
fi

# Iterate over all .sh files in the directory and execute them sequentially
for script in "$SCRIPT_DIR"/*.sh; do
    if [ -f "$script" ]; then
        echo "========================================================"
        echo "▶ Starting: $script"
        echo "▶ Start time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================================"
        
        # Execute the script (using 'bash' ensures it runs even if the file lacks execute permissions)
        bash "$script"
        
        echo "========================================================"
        echo "✔ Finished: $script"
        echo "✔ End time: $(date '+%Y-%m-%d %H:%M:%S')"
        echo "========================================================"
        echo ""
    fi
done

echo "🎉 All scripts have been executed successfully!"