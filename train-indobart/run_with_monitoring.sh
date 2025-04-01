#!/bin/bash

# Script to run IndoBART training with monitoring
# This script will keep your server alive while running the training process

# Exit on error
set -e

# Setup
TRAINING_SCRIPT="./run_all.sh"
OUTPUT_DIR="./indobart-pretrained"
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"

# Make script executable
chmod +x training_monitor.py

echo "========================================"
echo "IndoBART Training with Server Monitoring"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "========================================"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Start the training process in background
echo "Starting training process..."
nohup bash $TRAINING_SCRIPT > $LOG_FILE 2>&1 &
TRAINING_PID=$!
echo "Training process started with PID: $TRAINING_PID"

# Give some time for the training to initialize
sleep 5

# Start the monitoring script
echo "Starting monitoring script..."
python training_monitor.py --output_dir $OUTPUT_DIR --check_interval 300 --keep_alive

# Instructions on how to check results
echo ""
echo "========================================"
echo "How to check training results:"
echo "========================================"
echo "1. View log file: cat $LOG_FILE"
echo "2. Monitor training status: cat training_monitor.log"
echo "3. Check for checkpoints: ls -l $OUTPUT_DIR/checkpoint-*"
echo "4. Check model files: ls -l $OUTPUT_DIR/*.bin"
echo "========================================"
