#!/bin/bash

# Script to run IndoBART training with critical fixes for the "stuck at 0%" issue
# This applies emergency fixes to resolve deeply rooted issues in the training process

# Exit on error
set -e

# Setup
TRAINING_SCRIPT="./run_all.sh"
OUTPUT_DIR="./indobart-pretrained"
LOG_FILE="training_fixed_$(date +%Y%m%d_%H%M%S).log"

# Install required packages
echo "Installing required packages..."
pip install psutil tqdm --quiet

# Make scripts executable
chmod +x training_monitor.py
chmod +x run_all.sh

echo "========================================"
echo "IndoBART Training with Critical Fixes"
echo "========================================"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo "========================================"

# Create output directory if it doesn't exist
mkdir -p $OUTPUT_DIR

# Apply critical fixes to run_pretraining.py
echo "Applying critical trainer fixes..."

# 1. Inject the trainer_fix import at the top of run_pretraining.py
TMP_FILE=$(mktemp)
cat > $TMP_FILE << 'EOF'
# CRITICAL FIX: Import trainer_fix to resolve "stuck at 0%" issue
import os
os.environ["WANDB_DISABLED"] = "true"  # Disable wandb which can cause hangs
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Disable tokenizers parallelism
os.environ["PYTHONUNBUFFERED"] = "1"  # Ensure Python output is not buffered

try:
    import trainer_fix
    trainer_fix.apply_fixes()
    print("✅ Successfully applied critical trainer fixes for 'stuck at 0%' issue")
except ImportError:
    print("⚠️ Could not import trainer_fix, continuing without fixes")
except Exception as e:
    print(f"⚠️ Error applying trainer fixes: {e}")

EOF

# Backup original file
cp run_pretraining.py run_pretraining.py.bak

# Add imports at the top
cat $TMP_FILE > run_pretraining.py.fixed
cat run_pretraining.py.bak >> run_pretraining.py.fixed
mv run_pretraining.py.fixed run_pretraining.py

# 2. Modify key configuration parameters in run_all.sh for better stability
echo "Optimizing training parameters..."
cp run_all.sh run_all.sh.bak
sed -i.bak2 's/PER_DEVICE_BATCH_SIZE=8/PER_DEVICE_BATCH_SIZE=2/' run_all.sh
sed -i.bak3 's/GRADIENT_ACCUMULATION_STEPS=4/GRADIENT_ACCUMULATION_STEPS=16/' run_all.sh
sed -i.bak4 's/LOGGING_STEPS=500/LOGGING_STEPS=10/' run_all.sh
sed -i.bak5 's/FP16=true/FP16=false/' run_all.sh
sed -i.bak6 's/MAX_SEQ_LENGTH=512/MAX_SEQ_LENGTH=128/' run_all.sh  # Shorter sequences for debugging

# Add aggressive timeout setting for optimizer
sed -i.bak7 '/LEARNING_RATE=/ a\
# Fix for stuck training\
export TORCH_DISTRIBUTED_TIMEOUT=600  # 10 minute timeout for distributed operations\
' run_all.sh

# 3. Create emergency debugging script to monitor the training process
cat > debug_training.py << 'EOF'
#!/usr/bin/env python
"""Emergency debugging script to monitor and report on training process."""
import os
import time
import psutil
import sys
import subprocess
import signal
import threading

REFRESH_INTERVAL = 10  # seconds

def get_process_info(name="python"):
    """Get info about all Python processes"""
    processes = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline', 'cpu_percent', 'memory_percent']):
        if name in ' '.join(proc.info['cmdline'] or []):
            processes.append({
                'pid': proc.info['pid'],
                'cpu': proc.info['cpu_percent'],
                'memory': proc.info['memory_percent'],
                'cmd': ' '.join(proc.info['cmdline'] or [])[:100]
            })
    return processes

def get_system_resources():
    """Get system resource usage"""
    return {
        'cpu': psutil.cpu_percent(),
        'memory': psutil.virtual_memory().percent,
        'swap': psutil.swap_memory().percent
    }

def check_log_file(log_file):
    """Check if training is making progress"""
    if not os.path.exists(log_file):
        return "Log file not found"
    
    try:
        # Get the last 10 lines of the log file
        output = subprocess.check_output(f"tail -n 10 {log_file}", shell=True).decode('utf-8')
        return output
    except:
        return "Could not read log file"

def emergency_handler(signum, frame):
    """Handler for emergency signal (Ctrl+C)"""
    print("\n⚠️ EMERGENCY HANDLER ACTIVATED ⚠️")
    print("Force killing all Python processes...")
    
    # Kill all Python processes
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if 'python' in ' '.join(proc.info['cmdline'] or []):
            try:
                if proc.pid != os.getpid():  # Don't kill ourselves
                    print(f"Killing PID {proc.pid}: {' '.join(proc.info['cmdline'] or [])[:50]}")
                    os.kill(proc.pid, signal.SIGKILL)
            except:
                pass
    
    print("Emergency exit completed. Training has been forcibly stopped.")
    sys.exit(0)

def main():
    # Register emergency handler
    signal.signal(signal.SIGINT, emergency_handler)
    
    log_file = sys.argv[1] if len(sys.argv) > 1 else "training_fixed.log"
    print(f"🔍 Emergency Debugging Monitor Started 🔍")
    print(f"Monitoring log file: {log_file}")
    print("Press Ctrl+C for emergency shutdown of all training processes")
    
    starting_time = time.time()
    last_check_time = starting_time
    
    # Monitor in a separate thread to allow for emergency handler
    def monitoring_loop():
        nonlocal last_check_time
        while True:
            current_time = time.time()
            elapsed = current_time - starting_time
            time_since_last = current_time - last_check_time
            
            # Print separator
            print("\n" + "="*80)
            print(f"⏱️  MONITOR UPDATE - Running for {elapsed:.1f}s - {time.strftime('%H:%M:%S')}")
            
            # System resources
            resources = get_system_resources()
            print(f"💻 System Resources: CPU {resources['cpu']}% | Memory {resources['memory']}% | Swap {resources['swap']}%")
            
            # Process information
            processes = get_process_info()
            print(f"🔄 Found {len(processes)} Python processes:")
            for i, proc in enumerate(processes):
                print(f"  {i+1}. PID {proc['pid']}: CPU {proc['cpu']}% | Memory {proc['memory']}% | {proc['cmd']}")
            
            # Check log file
            if time_since_last > 30:  # Only check log every 30 seconds
                print("\n📋 Recent log entries:")
                log_tail = check_log_file(log_file)
                print(log_tail)
                last_check_time = current_time
            
            # Sleep and prepare for next update
            time.sleep(REFRESH_INTERVAL)
    
    # Start monitoring in a separate thread
    monitor_thread = threading.Thread(target=monitoring_loop)
    monitor_thread.daemon = True
    monitor_thread.start()
    
    # Keep main thread alive to handle Ctrl+C
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        # Let the emergency handler deal with this
        pass

if __name__ == "__main__":
    main()
EOF

chmod +x debug_training.py

echo "✅ Fixes applied successfully. Ready to start training."

# Start the training process with all fixes
echo "Starting training process with critical fixes..."

# Export critical environment variables
export WANDB_DISABLED=true
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TORCH_DISTRIBUTED_TIMEOUT=600

# Start training with complete logging
nohup bash -c "time bash $TRAINING_SCRIPT" > $LOG_FILE 2>&1 &
TRAINING_PID=$!
echo "Training process started with PID: $TRAINING_PID"

# Wait a moment to let it initialize
sleep 5

# Start the monitoring script in a separate terminal
echo "Starting emergency debugging monitor..."
python debug_training.py $LOG_FILE &
MONITOR_PID=$!

# Also start the regular monitoring
python training_monitor.py --output_dir $OUTPUT_DIR --check_interval 60 --keep_alive --log_file $LOG_FILE &
REGULAR_MONITOR_PID=$!

echo ""
echo "========================================"
echo "Training has been started with critical fixes."
echo "If training is still stuck after 30 minutes, try these steps:"
echo ""
echo "1. Press Ctrl+C in the debug monitor terminal to force kill all training processes"
echo "2. Try an even smaller model configuration:"
echo "   - Edit run_all.sh to use MAX_SAMPLES=100000 (10% of data)"
echo "   - Use a smaller model: facebook/bart-base -> distilbart-12-6"
echo "3. Check the log for specific errors: cat $LOG_FILE | grep -i error"
echo ""
echo "Log files:"
echo "- Main training log: $LOG_FILE"
echo "- Trainer fix log: trainer_fix.log"
echo "- Monitor log: training_monitor.log"
echo "========================================"

# Wait for user input
read -p "Press Enter to stop monitoring (training will continue in background)..."

# Kill monitoring processes but keep training running
kill $MONITOR_PID 2>/dev/null || true
kill $REGULAR_MONITOR_PID 2>/dev/null || true

echo "Monitoring stopped. Training continues in background."
echo "To view progress: tail -f $LOG_FILE"
