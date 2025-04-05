#!/bin/bash

# --- Configuration ---
# !!! IMPORTANT: Set these paths according to your server environment !!!

# Path to the directory containing the final pre-trained model and tokenizer
# (output of run_pretraining.sh)
# Example: MODEL_PATH="./indobart-pretrained"
MODEL_PATH="./indobart-pretrained"

# Path to the test text file (should be separate from training data)
# Example: TEST_FILE="./data/test_oscar_id.txt"
TEST_FILE="./data/test_oscar_id.txt" # Make sure this file exists and contains held-out data

# --- Evaluation Parameters ---
BATCH_SIZE=16 # Adjust based on GPU memory
MAX_SEQ_LENGTH=512

# --- Script Execution ---
echo "Starting Perplexity Evaluation..."
echo "Model path: ${MODEL_PATH}"
echo "Test file: ${TEST_FILE}"

# Check if test file exists
if [ ! -f "${TEST_FILE}" ]; then
    echo "Error: Test file not found at ${TEST_FILE}"
    exit 1
fi

# Run the Python evaluation script
python evaluate_perplexity.py \
    --model_path "${MODEL_PATH}" \
    --test_file "${TEST_FILE}" \
    --batch_size ${BATCH_SIZE} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    # Add --device cpu if you want to force CPU evaluation

# Check exit status
if [ $? -eq 0 ]; then
    echo "Evaluation script finished successfully."
else
    echo "Evaluation script failed. Check logs for details."
    exit 1
fi

echo "Evaluation complete."
