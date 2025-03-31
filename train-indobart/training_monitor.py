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

def monitor_training(args):
    """Main monitoring function"""
    logger.info(f"Starting training monitor for output directory: {args.output_dir}")
    logger.info(f"Check interval: {args.check_interval} seconds")
    
    # Last time we saw a checkpoint update
    last_update_time = time.time()
    last_checkpoint = None
    
    while True:
        current_time = time.time()
        stats = get_training_stats(args.output_dir)
        
        # Check if there are new checkpoints
        if stats["last_checkpoint"] != last_checkpoint:
            if last_checkpoint is not None:
                logger.info(f"New checkpoint created: {stats['last_checkpoint']}")
            last_checkpoint = stats["last_checkpoint"]
            last_update_time = current_time
        
        # Check training process
        training_running = check_process_running("run_pretraining.py")
        
        # Log current status
        logger.info("====== Training Status Update ======")
        logger.info(f"Training process running: {training_running}")
        if stats["last_checkpoint"]:
            logger.info(f"Latest checkpoint: {stats['last_checkpoint']} (created: {stats['checkpoint_time']})")
            logger.info(f"Total checkpoints: {stats['checkpoint_count']}")
        else:
            logger.info("No checkpoints found yet")
        
        logger.info(f"Log files found: {len(stats['log_files'])}")
        logger.info(f"TensorBoard files found: {len(stats['tensorboard_files'])}")
        logger.info(f"Model files found: {len(stats['model_files'])}")
        
        # Check if training appears to be complete or stalled
        time_since_update = current_time - last_update_time
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
