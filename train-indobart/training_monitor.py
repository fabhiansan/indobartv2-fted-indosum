#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Training Monitor for IndoBART
Keeps server alive and logs training progress
"""

import os
import time
import json
import glob
import logging
import argparse
import subprocess
import psutil
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training_monitor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("Training Monitor")

def parse_args():
    parser = argparse.ArgumentParser(description="Monitor training process and keep server alive")
    parser.add_argument("--output_dir", type=str, default="./indobart-pretrained",
                       help="Directory where model outputs are saved")
    parser.add_argument("--check_interval", type=int, default=300,
                       help="Interval in seconds between checks (default: 300s = 5min)")
    parser.add_argument("--keep_alive", action="store_true",
                       help="Keep the script running even if training seems complete")
    parser.add_argument("--monitor_resources", action="store_true", default=True,
                       help="Monitor system resources during training")
    parser.add_argument("--log_file", type=str, default=None,
                       help="Path to training log file to monitor")
    return parser.parse_args()

def get_latest_checkpoint(output_dir):
    """Get the latest checkpoint directory and its creation time"""
    checkpoint_dirs = glob.glob(os.path.join(output_dir, "checkpoint-*"))
    if not checkpoint_dirs:
        return None, None
    
    # Get the latest checkpoint based on creation time
    latest = max(checkpoint_dirs, key=os.path.getctime)
    creation_time = datetime.fromtimestamp(os.path.getctime(latest))
    
    return latest, creation_time

def get_log_files(output_dir):
    """Get all log files in the output directory"""
    log_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.endswith('.log') or file.endswith('.txt') or 'log' in file:
                log_files.append(os.path.join(root, file))
    return log_files

def check_tensorboard_logs(output_dir):
    """Check for TensorBoard event files"""
    tb_files = []
    for root, _, files in os.walk(output_dir):
        for file in files:
            if file.startswith('events.out.tfevents'):
                tb_files.append(os.path.join(root, file))
    return tb_files

def get_system_resources():
    """Get current system resource usage"""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Try to get GPU info if available
    gpu_info = "Not available"
    try:
        gpu_output = subprocess.check_output('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader', 
                                            shell=True).decode('utf-8').strip()
        gpu_info = gpu_output
    except:
        pass
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_used_gb": round(memory.used / (1024**3), 2),
        "memory_total_gb": round(memory.total / (1024**3), 2),
        "disk_percent": disk.percent,
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "gpu_info": gpu_info
    }

def check_training_log_progress(log_file):
    """Check training log for progress updates"""
    if not log_file or not os.path.exists(log_file):
        return {"status": "Log file not found"}
    
    try:
        # Get the last 20 lines of the log file
        output = subprocess.check_output(f"tail -n 20 {log_file}", shell=True).decode('utf-8')
        
        # Check for progress indicators
        progress_info = {}
        
        # Look for progress bar
        progress_lines = [line for line in output.split('\n') if '%|' in line]
        if progress_lines:
            progress_info["progress_bar"] = progress_lines[-1].strip()
            
            # Extract percentage
            try:
                percent = progress_lines[-1].split('%|')[0].strip()
                progress_info["percent_complete"] = percent
            except:
                pass
        
        # Look for loss values
        loss_lines = [line for line in output.split('\n') if 'loss' in line.lower()]
        if loss_lines:
            progress_info["latest_loss"] = loss_lines[-1].strip()
        
        # Check if we're still initializing
        if "Running training" in output and not progress_lines:
            progress_info["status"] = "Initializing"
        elif progress_lines:
            progress_info["status"] = "Training"
        else:
            progress_info["status"] = "Unknown"
            
        return progress_info
    except Exception as e:
        return {"status": f"Error reading log: {str(e)}"}

def get_training_stats(output_dir):
    """Get training statistics from the output directory"""
    stats = {
        "last_checkpoint": None,
        "checkpoint_time": None,
        "checkpoint_count": 0,
        "log_files": [],
        "tensorboard_files": [],
        "model_files": []
    }
    
    # Check if output directory exists
    if not os.path.exists(output_dir):
        logger.warning(f"Output directory {output_dir} does not exist yet.")
        return stats
    
    # Get checkpoint information
    latest_checkpoint, creation_time = get_latest_checkpoint(output_dir)
    if latest_checkpoint:
        stats["last_checkpoint"] = os.path.basename(latest_checkpoint)
        stats["checkpoint_time"] = creation_time.strftime("%Y-%m-%d %H:%M:%S")
        stats["checkpoint_count"] = len(glob.glob(os.path.join(output_dir, "checkpoint-*")))
    
    # Get log files
    stats["log_files"] = get_log_files(output_dir)
    
    # Get TensorBoard files
    stats["tensorboard_files"] = check_tensorboard_logs(output_dir)
    
    # Get model files (.bin, .json, etc.)
    model_files = []
    for ext in ['.bin', '.json', '.pt', '.model']:
        model_files.extend(glob.glob(os.path.join(output_dir, f"*{ext}")))
    stats["model_files"] = model_files
    
    return stats

def check_process_running(process_name="run_pretraining.py"):
    """Check if the training process is running"""
    try:
        output = subprocess.check_output(f"ps aux | grep {process_name} | grep -v grep", shell=True)
        return bool(output.strip())
    except subprocess.CalledProcessError:
        return False

def get_process_info(process_name="run_pretraining.py"):
    """Get detailed info about the training process"""
    try:
        # Get process IDs
        ps_output = subprocess.check_output(f"ps aux | grep {process_name} | grep -v grep", shell=True).decode('utf-8')
        lines = ps_output.strip().split('\n')
        
        processes = []
        for line in lines:
            parts = line.split()
            if len(parts) > 10:
                user = parts[0]
                pid = parts[1]
                cpu = parts[2]
                mem = parts[3]
                start_time = parts[9]
                cmd = ' '.join(parts[10:])
                
                processes.append({
                    "user": user,
                    "pid": pid,
                    "cpu_percent": cpu,
                    "mem_percent": mem,
                    "start_time": start_time,
                    "command": cmd[:50] + "..." if len(cmd) > 50 else cmd
                })
        
        return processes
    except Exception as e:
        return [{"error": str(e)}]

def debug_initialization_stuck(log_file):
    """Try to debug why initialization is stuck"""
    if not log_file or not os.path.exists(log_file):
        return "Cannot debug - log file not found"
    
    try:
        # Get all lines with timestamps in the last hour
        output = subprocess.check_output(f"grep -a '`date +%H:`' {log_file} | tail -n 50", shell=True).decode('utf-8')
        last_lines = output.strip().split('\n')
        
        # Look for specific patterns that might indicate issues
        debug_info = []
        
        # Check if we're stuck at dataset loading
        dataset_lines = [line for line in last_lines if "dataset" in line.lower() or "loading" in line.lower()]
        if dataset_lines:
            debug_info.append(f"Dataset related activity: {dataset_lines[-1]}")
        
        # Check for any error messages
        error_lines = [line for line in last_lines if "error" in line.lower() or "exception" in line.lower() or "fail" in line.lower()]
        if error_lines:
            debug_info.append(f"Possible error detected: {error_lines[-1]}")
        
        # Check if we started but are slow
        step_lines = [line for line in last_lines if "step" in line.lower() and "it/s" in line.lower()]
        if step_lines:
            debug_info.append(f"Training progress detected: {step_lines[-1]}")
        
        # Check memory related info
        memory_lines = [line for line in last_lines if "memory" in line.lower() or "ram" in line.lower() or "gpu" in line.lower()]
        if memory_lines:
            debug_info.append(f"Memory related info: {memory_lines[-1]}")
        
        if not debug_info:
            # If no specific issues found, get the last few lines
            try:
                tail_output = subprocess.check_output(f"tail -n 5 {log_file}", shell=True).decode('utf-8')
                debug_info.append(f"Last log entries:\n{tail_output}")
            except:
                debug_info.append("Could not get last log entries")
        
        return "\n".join(debug_info)
    except Exception as e:
        return f"Debug error: {str(e)}"

def suggest_fixes(log_file, output_dir):
    """Suggest potential fixes based on log analysis"""
    if not log_file or not os.path.exists(log_file):
        return ["Create detailed logs by redirecting stdout and stderr"]
    
    suggestions = []
    
    # Check system resources
    resources = get_system_resources()
    if resources["memory_percent"] > 90:
        suggestions.append("System is low on memory. Consider reducing batch size or using gradient checkpointing.")
    
    if resources["cpu_percent"] > 95:
        suggestions.append("CPU usage is very high. Consider reducing the number of DataLoader workers.")
    
    # Check log for specific issues
    try:
        log_content = subprocess.check_output(f"grep -a 'error\\|warning\\|cuda\\|memory\\|batch' {log_file} | tail -n 20", 
                                             shell=True).decode('utf-8')
        
        if "CUDA out of memory" in log_content:
            suggestions.append("CUDA out of memory error detected. Reduce batch size and/or model size.")
            
        if "killed" in log_content.lower():
            suggestions.append("Process may have been killed by the OS. This often happens due to memory limits.")
            
        if "batch" in log_content.lower() and "error" in log_content.lower():
            suggestions.append("Possible issue with batch processing. Check for NaN values or dataset problems.")
            
        # Check initialization time
        first_log = subprocess.check_output(f"head -n 20 {log_file}", shell=True).decode('utf-8')
        if "Running training" in first_log and "Num examples" in first_log:
            suggestions.append("Training initialization successful but progress stuck. Wait longer for first batch or check for resource bottlenecks.")
            
    except Exception as e:
        suggestions.append(f"Could not analyze log file: {str(e)}")
    
    # General suggestions
    if not os.path.exists(os.path.join(output_dir, "checkpoint-0")):
        suggestions.append("No checkpoints found. Try reducing batch size and increasing logging frequency.")
    
    suggestions.append("Add print statements in run_pretraining.py before the training loop to track progress")
    suggestions.append("Monitor GPU memory usage with 'nvidia-smi' in a separate terminal")
    
    return suggestions

def monitor_training(args):
    """Main monitoring function"""
    logger.info(f"Starting training monitor for output directory: {args.output_dir}")
    logger.info(f"Check interval: {args.check_interval} seconds")
    
    # Last time we saw a checkpoint update
    last_update_time = time.time()
    last_checkpoint = None
    stuck_counter = 0
    
    while True:
        current_time = time.time()
        stats = get_training_stats(args.output_dir)
        
        # Check if there are new checkpoints
        if stats["last_checkpoint"] != last_checkpoint:
            if last_checkpoint is not None:
                logger.info(f"New checkpoint created: {stats['last_checkpoint']}")
            last_checkpoint = stats["last_checkpoint"]
            last_update_time = current_time
            stuck_counter = 0
        
        # Check training process
        training_running = check_process_running("run_pretraining.py")
        process_info = get_process_info("run_pretraining.py") if training_running else []
        
        # Check training log progress if available
        training_progress = {}
        if args.log_file:
            training_progress = check_training_log_progress(args.log_file)
        
        # Monitor system resources if enabled
        resource_info = {}
        if args.monitor_resources:
            resource_info = get_system_resources()
        
        # Log current status
        logger.info("====== Training Status Update ======")
        logger.info(f"Training process running: {training_running}")
        
        if process_info:
            logger.info("Process information:")
            for proc in process_info:
                for key, value in proc.items():
                    logger.info(f"  {key}: {value}")
        
        if resource_info:
            logger.info("System resources:")
            for key, value in resource_info.items():
                logger.info(f"  {key}: {value}")
        
        if training_progress:
            logger.info("Training progress:")
            for key, value in training_progress.items():
                logger.info(f"  {key}: {value}")
        
        if stats["last_checkpoint"]:
            logger.info(f"Latest checkpoint: {stats['last_checkpoint']} (created: {stats['checkpoint_time']})")
            logger.info(f"Total checkpoints: {stats['checkpoint_count']}")
        else:
            logger.info("No checkpoints found yet")
        
        logger.info(f"Log files found: {len(stats['log_files'])}")
        logger.info(f"TensorBoard files found: {len(stats['tensorboard_files'])}")
        logger.info(f"Model files found: {len(stats['model_files'])}")
        
        # Check if training appears to be stuck
        time_since_update = current_time - last_update_time
        if training_running and time_since_update > args.check_interval * 3:
            stuck_counter += 1
            if stuck_counter >= 3:  # If stuck for 3 consecutive checks
                logger.warning("Training appears to be stuck!")
                
                # Try to debug why it's stuck
                if args.log_file:
                    debug_info = debug_initialization_stuck(args.log_file)
                    logger.warning(f"Debug information:\n{debug_info}")
                    
                    # Suggest fixes
                    fixes = suggest_fixes(args.log_file, args.output_dir)
                    logger.warning("Suggested fixes:")
                    for i, fix in enumerate(fixes, 1):
                        logger.warning(f"{i}. {fix}")
                
                # Reset counter so we don't spam this warning
                stuck_counter = 0
        
        # Check if training appears to be complete or stalled
        if not training_running and not args.keep_alive:
            if time_since_update > args.check_interval * 3:  # No updates for 3x check interval
                logger.info("Training process is not running and no recent updates detected.")
                logger.info("You can restart the monitor with --keep_alive to continue monitoring.")
                break
        
        # Keep the script alive
        logger.info(f"Next check in {args.check_interval} seconds...")
        time.sleep(args.check_interval)

def main():
    args = parse_args()
    try:
        monitor_training(args)
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    except Exception as e:
        logger.error(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    main()
