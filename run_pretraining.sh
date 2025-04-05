#!/bin/bash

# --- Configuration ---
# !!! IMPORTANT: Set these paths and hyperparameters according to your server environment and desired training setup !!!

# Path to the trained tokenizer directory (output of run_tokenizer_training.sh)
# Example: TOKENIZER_PATH="./indobart-tokenizer"
TOKENIZER_PATH="./indobart-tokenizer"

# Path to the cleaned text file (output of run_preprocessing.sh)
# Example: DATA_FILE="./data/cleaned_oscar_id.txt"
DATA_FILE="./data/cleaned_oscar_id.txt"

# Directory where the final trained model and checkpoints will be saved
# Example: OUTPUT_DIR="./indobart-pretrained"
OUTPUT_DIR="./indobart-pretrained"

# Base BART config name (e.g., facebook/bart-base)
CONFIG_NAME="facebook/bart-base"

# --- Training Hyperparameters ---
# Adjust based on your GPU memory and desired effective batch size
PER_DEVICE_TRAIN_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4 # Effective batch size = N_GPUS * PER_DEVICE_TRAIN_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS

LEARNING_RATE=5e-5
WEIGHT_DECAY=0.01
ADAM_BETA1=0.9
ADAM_BETA2=0.999
ADAM_EPSILON=1e-8
MAX_GRAD_NORM=1.0
NUM_TRAIN_EPOCHS=3 # Adjust based on dataset size and convergence
LR_SCHEDULER_TYPE="linear" # e.g., linear, cosine
WARMUP_STEPS=500 # Number of warmup steps for learning rate scheduler

# --- Logging/Saving ---
LOGGING_STEPS=100 # Log metrics every N steps
SAVE_STEPS=1000   # Save checkpoint every N steps
SAVE_TOTAL_LIMIT=2 # Keep only the last N checkpoints + the final model

# --- Distributed Training (using accelerate) ---
# Set NUM_PROCESSES to the number of GPUs you want to use
NUM_PROCESSES=1 # Change to > 1 for multi-GPU

# --- Script Execution ---
echo "Starting BART Pre-training..."
echo "Tokenizer path: ${TOKENIZER_PATH}"
echo "Data file: ${DATA_FILE}"
echo "Output directory: ${OUTPUT_DIR}"
echo "Num GPUs: ${NUM_PROCESSES}"
echo "Batch size per GPU: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "Gradient accumulation steps: ${GRADIENT_ACCUMULATION_STEPS}"
echo "Effective batch size: $((${NUM_PROCESSES} * ${PER_DEVICE_TRAIN_BATCH_SIZE} * ${GRADIENT_ACCUMULATION_STEPS}))"

# Ensure the output directory exists
mkdir -p "${OUTPUT_DIR}"

# Use accelerate launch for easy handling of single/multi-GPU and distributed training
# If accelerate is not installed: pip install accelerate
# You might need to run 'accelerate config' once on your server to set up defaults
accelerate launch run_pretraining.py \
    --model_output_dir "${OUTPUT_DIR}" \
    --tokenizer_path "${TOKENIZER_PATH}" \
    --config_name "${CONFIG_NAME}" \
    --cleaned_text_file "${DATA_FILE}" \
    --do_train \
    --output_dir "${OUTPUT_DIR}" \
    --per_device_train_batch_size ${PER_DEVICE_TRAIN_BATCH_SIZE} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --learning_rate ${LEARNING_RATE} \
    --weight_decay ${WEIGHT_DECAY} \
    --adam_beta1 ${ADAM_BETA1} \
    --adam_beta2 ${ADAM_BETA2} \
    --adam_epsilon ${ADAM_EPSILON} \
    --max_grad_norm ${MAX_GRAD_NORM} \
    --num_train_epochs ${NUM_TRAIN_EPOCHS} \
    --lr_scheduler_type "${LR_SCHEDULER_TYPE}" \
    --warmup_steps ${WARMUP_STEPS} \
    --logging_steps ${LOGGING_STEPS} \
    --save_steps ${SAVE_STEPS} \
    --save_total_limit ${SAVE_TOTAL_LIMIT} \
    --fp16 \
    --preprocessing_num_workers $(nproc) \
    --overwrite_cache false \
    --max_seq_length 512 \
    # Add --report_to tensorboard (or wandb) if needed for logging

# Check exit status
if [ $? -eq 0 ]; then
    echo "Pre-training script finished successfully."
else
    echo "Pre-training script failed. Check logs for details."
    exit 1
fi

echo "Pre-training complete. Model saved to ${OUTPUT_DIR}"

# --- Notes on Environment Setup ---
# Required libraries:
# pip install torch torchvision torchaudio
# pip install transformers datasets tokenizers accelerate sentencepiece
# Ensure CUDA is correctly installed if using GPUs.
