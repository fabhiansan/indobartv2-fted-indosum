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
    
    read -p "Do you want to continue and potentially overwrite existing data? (y/n): " choice
    case "$choice" in
        y|Y ) echo "Continuing with the script...";;
        * ) echo "Operation cancelled by user."; exit 0;;
    esac
fi

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$TOKENIZER_DIR"
mkdir -p "$OUTPUT_DIR"

# --- Step 1: Prepare Data ---
echo "--- Preparing Data --- "
python prepare_data.py \
    --corpus_name "$CORPUS_NAME" \
    --corpus_subset "$CORPUS_SUBSET" \
    --output_dir "$DATA_DIR" \
    --output_filename "$(basename $PREPARED_DATA_FILE)" \
    ${MAX_SAMPLES:+ --max_samples $MAX_SAMPLES} # Add max_samples only if set

echo "--- Data Preparation Complete. Output: $PREPARED_DATA_FILE ---"

# Check that the data file exists and is not empty
if [ ! -f "$PREPARED_DATA_FILE" ]; then
    echo "Error: Data file $PREPARED_DATA_FILE does not exist. Data preparation failed."
    exit 1
fi

if [ ! -s "$PREPARED_DATA_FILE" ]; then
    echo "Error: Data file $PREPARED_DATA_FILE is empty. Data preparation may have failed."
    exit 1
fi

echo "Data file check passed: $PREPARED_DATA_FILE exists and is not empty."
echo "File size: $(du -h "$PREPARED_DATA_FILE" | cut -f1)"

# --- Step 2: Train Tokenizer ---
echo "--- Training Tokenizer --- "
python train_tokenizer.py \
    --input_files "$PREPARED_DATA_FILE" \
    --output_dir "$TOKENIZER_DIR" \
    --vocab_size $VOCAB_SIZE \
    --min_frequency $MIN_FREQUENCY

echo "--- Tokenizer Training Complete. Output: $TOKENIZER_DIR ---"

# Check that the tokenizer files exist
if [ ! -f "$TOKENIZER_DIR/vocab.json" ] || [ ! -f "$TOKENIZER_DIR/merges.txt" ]; then
    echo "Error: Tokenizer files (vocab.json and merges.txt) not found in $TOKENIZER_DIR."
    echo "Tokenizer training may have failed."
    exit 1
fi

echo "Tokenizer check passed: vocab.json and merges.txt found in $TOKENIZER_DIR"

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

# Check if CUDA is available for GPU training
CUDA_AVAILABLE=false
if python -c "import torch; print(torch.cuda.is_available())" | grep -q "True"; then
    CUDA_AVAILABLE=true
    NUM_GPUS=$(python -c "import torch; print(torch.cuda.device_count())")
    echo "CUDA is available. Using $NUM_GPUS GPU(s)."
else
    echo "CUDA is not available. Using CPU for training (this will be slow)."
    # Disable FP16 if using CPU
    FP16=false
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
    --preprocessing_num_workers $(nproc) \
    --seed 42 \
    --bart_objective true \
    $(if [ "$FP16" = true ]; then echo "--fp16"; fi) # Add --fp16 only if FP16 is true
    # --line_by_line # Uncomment if you prefer line-by-line processing instead of grouping

# Check if training completed successfully
if [ ! -f "$OUTPUT_DIR/pytorch_model.bin" ]; then
    echo "Warning: Model checkpoint (pytorch_model.bin) not found in $OUTPUT_DIR."
    echo "Pre-training may have failed or is not yet complete."
    exit 1
fi

echo "--- Pre-training Complete. Model saved in: $OUTPUT_DIR ---"
echo "Model files:"
ls -lh "$OUTPUT_DIR"
