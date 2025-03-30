#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Error handling for cleaner error reporting
error_handler() {
    echo "Error occurred in line $1"
    exit 1
}
trap 'error_handler $LINENO' ERR

# --- Configuration ---
# Use a relative path based on user's home directory for HuggingFace cache
export HF_HOME="$HOME/.cache/huggingface" # Default location, can be changed
export TOKENIZERS_PARALLELISM=true # Enable fast tokenizer parallelism

# Data parameters
DATA_DIR="./data"
PREPARED_DATA_FILE="$DATA_DIR/indonesian.txt"
CORPUS_NAME="oscar"
CORPUS_SUBSET="unshuffled_deduplicated_id"
MAX_SAMPLES=1000000 # Optional: Limit dataset size for testing (e.g., 1000000). Remove or set to large number for full run.

# Cache parameters
CACHE_DIR="$DATA_DIR/cache"  # Directory to store cached datasets and preprocessed data
REUSE_CACHE=true  # Set to false to force regenerate all data even if cached
FORCE_RELOAD=false  # Set to true to force reload datasets from source

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

# --- Confirmation if overwriting existing data ---
if [ -d "$DATA_DIR" ] || [ -d "$TOKENIZER_DIR" ] || [ -d "$OUTPUT_DIR" ]; then
    echo "The following directories already exist:"
    [ -d "$DATA_DIR" ] && echo "- $DATA_DIR"
    [ -d "$TOKENIZER_DIR" ] && echo "- $TOKENIZER_DIR"
    [ -d "$OUTPUT_DIR" ] && echo "- $OUTPUT_DIR"
    
    read -p "Do you want to continue? Existing files may be used from cache if enabled. (y/n): " CONFIRM
    if [[ "$CONFIRM" != "y" && "$CONFIRM" != "Y" ]]; then
        echo "Operation aborted by user."
        exit 0
    fi
fi

# Make sure all necessary directories exist
mkdir -p "$DATA_DIR"
mkdir -p "$CACHE_DIR"
mkdir -p "$TOKENIZER_DIR"
mkdir -p "$OUTPUT_DIR"

# --- Step 1: Download and prepare the dataset ---
echo "Step 1: Preparing dataset..."

# Prepare cache arguments based on settings
CACHE_ARGS=""
if [ "$REUSE_CACHE" = true ]; then
    CACHE_ARGS="$CACHE_ARGS --reuse_cache"
    echo "Reuse cache enabled: Will use existing data files if available"
fi

if [ "$FORCE_RELOAD" = true ]; then
    CACHE_ARGS="$CACHE_ARGS --force_reload"
    echo "Force reload enabled: Will reload data from source"
fi

# Prepare the data (this will download the corpus if needed)
python prepare_data.py \
    --corpus_name "$CORPUS_NAME" \
    --corpus_subset "$CORPUS_SUBSET" \
    --output_dir "$DATA_DIR" \
    --output_filename "$(basename "$PREPARED_DATA_FILE")" \
    --cache_dir "$CACHE_DIR" \
    --max_samples "$MAX_SAMPLES" \
    $CACHE_ARGS

# Validate the prepared data exists
if [ ! -f "$PREPARED_DATA_FILE" ]; then
    echo "Error: Prepared data file not found at $PREPARED_DATA_FILE"
    exit 1
fi

echo "Dataset preparation complete. File saved to $PREPARED_DATA_FILE"
FILESIZE=$(du -h "$PREPARED_DATA_FILE" | cut -f1)
echo "File size: $FILESIZE"

# --- Step 2: Train the tokenizer ---
echo -e "\nStep 2: Training tokenizer..."

# Check if tokenizer already exists and is valid
if [ -f "$TOKENIZER_DIR/vocab.json" ] && [ -f "$TOKENIZER_DIR/merges.txt" ] && [ "$REUSE_CACHE" = true ]; then
    echo "Existing tokenizer found. Skipping tokenizer training as cache reuse is enabled."
else
    echo "Training new tokenizer..."
    python train_tokenizer.py \
        --input_file "$PREPARED_DATA_FILE" \
        --output_dir "$TOKENIZER_DIR" \
        --vocab_size "$VOCAB_SIZE" \
        --min_frequency "$MIN_FREQUENCY"
    
    # Validate the tokenizer files exist
    if [ ! -f "$TOKENIZER_DIR/vocab.json" ] || [ ! -f "$TOKENIZER_DIR/merges.txt" ]; then
        echo "Error: Tokenizer files not found after training"
        exit 1
    fi
fi

echo "Tokenizer ready at $TOKENIZER_DIR"

# --- Step 3: Pre-train the BART model ---
echo -e "\nStep 3: Pre-training the BART model..."

# Count available GPUs
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected $GPU_COUNT GPU(s)"
else
    GPU_COUNT=0
    echo "No NVIDIA GPUs detected"
fi

# Set up distributed training command based on GPU count
if [ $GPU_COUNT -gt 1 ] && command -v accelerate &> /dev/null; then
    LAUNCH_CMD="accelerate launch"
    echo "Using Accelerate for multi-GPU training ($GPU_COUNT GPUs)"
else
    LAUNCH_CMD="python"
    echo "Running in single device mode"
fi

# Add cache arguments for pretraining
PRETRAINING_CACHE_ARGS=""
if [ "$REUSE_CACHE" = true ]; then
    PRETRAINING_CACHE_ARGS="$PRETRAINING_CACHE_ARGS --use_cached_prep --dataset_cache_dir $CACHE_DIR"
    echo "Using cached preprocessed datasets if available"
fi

if [ "$FORCE_RELOAD" = true ]; then
    PRETRAINING_CACHE_ARGS="$PRETRAINING_CACHE_ARGS --force_reload_raw"
    echo "Forcing reload of raw datasets"
fi

# Confirm train file exists before starting
if [ ! -f "$PREPARED_DATA_FILE" ]; then
    echo "ERROR: Train file not found at $PREPARED_DATA_FILE"
    exit 1
else
    echo "Confirmed train file exists at $PREPARED_DATA_FILE ($(du -h "$PREPARED_DATA_FILE" | cut -f1))"
fi

# Start pretraining
$LAUNCH_CMD run_pretraining.py \
    --model_name_or_path "$BASE_MODEL" \
    --tokenizer_name_or_path "$TOKENIZER_DIR" \
    --train_file "$PREPARED_DATA_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --overwrite_output_dir \
    --do_train \
    --do_eval \
    --max_seq_length "$MAX_SEQ_LENGTH" \
    --per_device_train_batch_size "$PER_DEVICE_BATCH_SIZE" \
    --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
    --learning_rate "$LEARNING_RATE" \
    --warmup_steps "$WARMUP_STEPS" \
    --num_train_epochs "$NUM_EPOCHS" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit 3 \
    --evaluation_strategy steps \
    --eval_steps "$EVAL_STEPS" \
    --logging_steps "$LOGGING_STEPS" \
    --fp16 "$FP16" \
    --load_best_model_at_end \
    --bart_objective true \
    --masking_fraction 0.3 \
    --poisson_lambda 3.0 \
    $PRETRAINING_CACHE_ARGS

echo -e "\nPre-training complete! Model saved to $OUTPUT_DIR"
echo "You can now use this model for fine-tuning on downstream tasks."
