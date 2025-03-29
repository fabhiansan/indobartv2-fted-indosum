#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
export HF_HOME=/path/to/your/huggingface_cache # Optional: Set cache directory for models/datasets
export TOKENIZERS_PARALLELISM=true # Enable fast tokenizer parallelism

# Data parameters
DATA_DIR="./data"
PREPARED_DATA_FILE="$DATA_DIR/indonesian.txt"
CORPUS_NAME="oscar"
CORPUS_SUBSET="unshuffled_deduplicated_id"
MAX_SAMPLES=1000000 # Optional: Limit dataset size for testing (e.g., 1000000). Remove or set to large number for full run.

# Tokenizer parameters
TOKENIZER_DIR="./tokenizer"
VOCAB_SIZE=50265
MIN_FREQUENCY=2

# Model parameters
BASE_MODEL="facebook/bart-base"
OUTPUT_DIR="./indobart-pretrained"

# Training parameters (Adjust based on your hardware)
NUM_EPOCHS=3
PER_DEVICE_BATCH_SIZE=8
GRADIENT_ACCUMULATION_STEPS=4 # Effective batch size = N_GPUS * PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
LEARNING_RATE=5e-5
WARMUP_STEPS=1000
SAVE_STEPS=10000
EVAL_STEPS=10000 # Evaluate periodically
LOGGING_STEPS=500
MAX_SEQ_LENGTH=512
FP16=true # Set to false if your hardware doesn't support FP16

# --- Step 1: Prepare Data ---
echo "--- Preparing Data --- "
python prepare_data.py \
    --corpus_name "$CORPUS_NAME" \
    --corpus_subset "$CORPUS_SUBSET" \
    --output_dir "$DATA_DIR" \
    --output_filename "$(basename $PREPARED_DATA_FILE)" \
    ${MAX_SAMPLES:+ --max_samples $MAX_SAMPLES} # Add max_samples only if set

echo "--- Data Preparation Complete. Output: $PREPARED_DATA_FILE ---"

# --- Step 2: Train Tokenizer ---
echo "--- Training Tokenizer --- "
python train_tokenizer.py \
    --input_files "$PREPARED_DATA_FILE" \
    --output_dir "$TOKENIZER_DIR" \
    --vocab_size $VOCAB_SIZE \
    --min_frequency $MIN_FREQUENCY

echo "--- Tokenizer Training Complete. Output: $TOKENIZER_DIR ---"

# --- Step 3: Run Pre-training ---
echo "--- Starting Pre-training --- "

# Determine if accelerate is available and use it
if command -v accelerate &> /dev/null
then
    echo "Using accelerate for distributed training (if applicable)..."
    ACCELERATE_CMD="accelerate launch"
else
    echo "accelerate not found. Running with python..."
    ACCELERATE_CMD="python"
fi

$ACCELERATE_CMD run_pretraining.py \
    --model_name_or_path "$BASE_MODEL" \
    --tokenizer_name_or_path "$TOKENIZER_DIR" \
    --train_file "$PREPARED_DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --do_train \
    --do_eval \
    --validation_split_percentage 5 \
    --per_device_train_batch_size $PER_DEVICE_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --learning_rate $LEARNING_RATE \
    --num_train_epochs $NUM_EPOCHS \
    --warmup_steps $WARMUP_STEPS \
    --logging_steps $LOGGING_STEPS \
    --save_steps $SAVE_STEPS \
    --eval_steps $EVAL_STEPS \
    --evaluation_strategy "steps" \
    --save_total_limit 5 \
    --max_seq_length $MAX_SEQ_LENGTH \
    --overwrite_output_dir \
    --bart_objective true \
    --preprocessing_num_workers $(nproc) \
    --seed 42 \
    $(if [ "$FP16" = true ]; then echo "--fp16"; fi) # Add --fp16 only if FP16 is true
    # --line_by_line # Uncomment if you prefer line-by-line processing instead of grouping

echo "--- Pre-training Complete. Model saved in: $OUTPUT_DIR ---"
