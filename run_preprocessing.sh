#!/bin/bash

# --- Configuration ---
# !!! IMPORTANT: Set these paths according to your server environment !!!

# Path to the specific OSCAR dataset cache directory for Indonesian
# Example: CACHE_DIR="$HOME/.cache/huggingface/datasets/oscar-corpus___oscar-2301/id"
CACHE_DIR="$HOME/.cache/huggingface/datasets/oscar-corpus___oscar-2301/id"

# Path where the cleaned output text file should be saved
# Example: OUTPUT_FILE="./data/cleaned_oscar_id.txt"
OUTPUT_FILE="./data/cleaned_oscar_id.txt"

# --- Script Execution ---
echo "Starting OSCAR data preprocessing..."
echo "Using cache directory: ${CACHE_DIR}"
echo "Outputting cleaned data to: ${OUTPUT_FILE}"

# Ensure the output directory exists
mkdir -p "$(dirname "${OUTPUT_FILE}")"

# Run the Python preprocessing script
python preprocess_data.py \
    --cache_path "${CACHE_DIR}" \
    --output_file "${OUTPUT_FILE}"

# Check exit status
if [ $? -eq 0 ]; then
    echo "Preprocessing script finished successfully."
else
    echo "Preprocessing script failed. Check logs for details."
    exit 1
fi

echo "Preprocessing complete. Cleaned data saved to ${OUTPUT_FILE}"
