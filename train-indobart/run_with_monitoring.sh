#!/bin/bash

# Script to run IndoBART training with monitoring
# This script will keep your server alive while running the training process

# Exit on error
set -e

# Setup
TRAINING_SCRIPT="./run_all.sh"
OUTPUT_DIR="./indobart-pretrained"
LOG_FILE="training_$(date +%Y%m%d_%H%M%S).log"
DEBUG_MODE=${1:-false}  # Pass 'debug' as first argument to enable debug mode

# Install psutil if not already installed
echo "Checking for required packages..."
python -c "import psutil" 2>/dev/null || pip install psutil

# Make scripts executable
chmod +x training_monitor.py

echo "========================================"
echo "IndoBART Training with Server Monitoring"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "Debug mode: $DEBUG_MODE"
echo "========================================"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Apply fix for "stuck at 0%" issue
if [ "$DEBUG_MODE" = "debug" ]; then
    echo "Applying optimizations for better performance and debugging..."
    
    # Reduce batch size for better memory management
    sed -i.bak 's/PER_DEVICE_BATCH_SIZE=8/PER_DEVICE_BATCH_SIZE=4/' $TRAINING_SCRIPT
    sed -i.bak 's/GRADIENT_ACCUMULATION_STEPS=4/GRADIENT_ACCUMULATION_STEPS=8/' $TRAINING_SCRIPT
    
    # Increase logging frequency to get more feedback
    sed -i.bak 's/LOGGING_STEPS=500/LOGGING_STEPS=10/' $TRAINING_SCRIPT
    
    # Disable fp16 which can cause issues
    sed -i.bak 's/FP16=true/FP16=false/' $TRAINING_SCRIPT
    
    echo "Reduced batch size, increased gradient accumulation steps,"
    echo "increased logging frequency, and disabled fp16 for better stability."
    
    echo "Setting environment variable to disable WANDB..."
    export WANDB_DISABLED="true"
    
    echo "Adding debug prints to run_pretraining.py..."
    # Add debug prints at key points in the script
    TMP_FILE=$(mktemp)
    awk '
    /if __name__ == "__main__":/ {
        print "# === DEBUG CODE START ==="
        print "def time_it(func):"
        print "    def wrapper(*args, **kwargs):"
        print "        start_time = time.time()"
        print "        print(f\"Starting {func.__name__} at {time.strftime(\"%H:%M:%S\")}\")"
        print "        result = func(*args, **kwargs)"
        print "        end_time = time.time()"
        print "        print(f\"Finished {func.__name__} in {end_time - start_time:.2f} seconds\")"
        print "        return result"
        print "    return wrapper"
        print ""
        print "# Monkey patch Trainer.train to add timing"
        print "from transformers import Trainer"
        print "original_train = Trainer.train"
        print "Trainer.train = time_it(original_train)"
        print "print(\"Added timing instrumentation to Trainer.train\")"
        print "# === DEBUG CODE END ==="
        print $0
    }
    /Running training/ {
        print $0
        print "        print(\"DEBUG: About to start training loop\")"
        print "        print(f\"DEBUG: Memory usage before training: {psutil.Process().memory_info().rss / (1024**2):.2f} MB\")"
        print "        import torch"
        print "        if torch.cuda.is_available():"
        print "            print(f\"DEBUG: GPU memory: {torch.cuda.memory_allocated() / (1024**3):.2f} GB / {torch.cuda.max_memory_allocated() / (1024**3):.2f} GB\")"
    }
    !/Running training/ && !/if __name__ == "__main__":/ {print}
    ' run_pretraining.py > $TMP_FILE
    mv $TMP_FILE run_pretraining.py
    
    # Add psutil import at the top if it's not there
    if ! grep -q "import psutil" run_pretraining.py; then
        TMP_FILE=$(mktemp)
        awk 'NR==1 {print "import psutil"; print} NR>1 {print}' run_pretraining.py > $TMP_FILE
        mv $TMP_FILE run_pretraining.py
    fi
    
    echo "Debug instrumentation added to run_pretraining.py"
fi

# Start the training process in background with verbose output
echo "Starting training process with detailed logging..."
if [ "$DEBUG_MODE" = "debug" ]; then
    # Run with more verbose output in debug mode
    nohup bash -c "
        export TOKENIZERS_PARALLELISM=false  # Disable parallelism which can cause hangs
        export WANDB_DISABLED=true           # Disable wandb which can slow things down
        export PYTHONUNBUFFERED=1            # Ensure Python output is not buffered
        time bash $TRAINING_SCRIPT 
    " > $LOG_FILE 2>&1 &
else
    # Standard run
    nohup bash $TRAINING_SCRIPT > $LOG_FILE 2>&1 &
fi

TRAINING_PID=$!
echo "Training process started with PID: $TRAINING_PID"

# Give some time for the training to initialize
sleep 5

# Start the monitoring script
echo "Starting monitoring script..."
python training_monitor.py --output_dir $OUTPUT_DIR --check_interval 60 --keep_alive --log_file $LOG_FILE

# Instructions on how to check results
echo ""
echo "========================================"
echo "How to check training results:"
echo "========================================"
echo "1. View log file: cat $LOG_FILE"
echo "2. Monitor training status: cat training_monitor.log"
echo "3. Check for checkpoints: ls -l $OUTPUT_DIR/checkpoint-*"
echo "4. Check model files: ls -l $OUTPUT_DIR/*.bin"
echo ""
echo "If training is stuck at 0%:"
echo "1. Run with debug mode: ./run_with_monitoring.sh debug"
echo "2. Check system resources: top"
echo "3. Look for errors: grep -i error $LOG_FILE"
echo "========================================"
