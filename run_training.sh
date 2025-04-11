#!/bin/bash

# Script to run the summarization fine-tuning process.

# Ensure you have activated the correct Python environment
# where the required libraries are installed.
# (e.g., conda activate my_env or source venv/bin/activate)

# Reminder: Install dependencies if you haven't already:
# pip install torch transformers datasets evaluate rouge_score nltk sentencepiece accelerate tensorboard

echo "Starting the fine-tuning script (run_finetuning.py)..."
echo "Check finetune_config.py for configuration settings."
echo "Logs will be printed to the console and saved to TensorBoard (./runs)."

# Execute the Python script
python run_finetuning.py

# Check the exit code of the Python script
EXIT_CODE=$?
if [ $EXIT_CODE -eq 0 ]; then
  echo "Fine-tuning script finished successfully."
else
  echo "Fine-tuning script exited with error code: $EXIT_CODE"
fi