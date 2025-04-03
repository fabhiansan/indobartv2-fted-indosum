#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Adjust these paths and parameters as needed for your environment

# Data
export OSCAR_DATASET_NAME="oscar-corpus/OSCAR-2301"
export OSCAR_DATASET_LANG="id"
export SUMMARIZATION_DATASET_NAME="indosum" # Or path to your local dataset like ./data/indosum
# export SUMMARIZATION_TRAIN_FILE="./data/indosum/train.json" # Example if using local files
# export SUMMARIZATION_VALIDATION_FILE="./data/indosum/validation.json" # Example
# export SUMMARIZATION_TEST_FILE="./data/indosum/test.json" # Example

# Tokenizer
export TOKENIZER_OUTPUT_DIR="./indonesian_tokenizer"
export TOKENIZER_VOCAB_SIZE=32000

# Pre-training
export BASE_MODEL="facebook/bart-base"
export PRETRAINED_OUTPUT_DIR="./indobart_pretrained"
export PRETRAINING_EPOCHS=1 # Adjust as needed (can be float)
export PRETRAINING_BATCH_SIZE=24 # Adjust based on GPU memory
export PRETRAINING_LR=5e-5

# Fine-tuning
export FINETUNED_OUTPUT_DIR="./indobart_summarizer"
export FINETUNING_EPOCHS=3 # Adjust as needed
export FINETUNING_BATCH_SIZE=24 # Adjust based on GPU memory
export FINETUNING_LR=3e-5

# --- Helper Function ---
run_step() {
  echo "--------------------------------------------------"
  echo "Running Step: $1"
  echo "Command: $2"
  echo "--------------------------------------------------"
  eval $2 # Execute the command
  if [ $? -ne 0 ]; then
    echo "Error: Step '$1' failed. Exiting."
    exit 1
  fi
  echo "--------------------------------------------------"
  echo "Step '$1' completed successfully."
  echo "--------------------------------------------------"
}

# --- Pipeline Steps ---

# 1. Train Tokenizer
STEP_NAME="Train Tokenizer"
COMMAND="python train_tokenizer.py \
  --dataset_name \"$OSCAR_DATASET_NAME\" \
  --dataset_lang \"$OSCAR_DATASET_LANG\" \
  --vocab_size $TOKENIZER_VOCAB_SIZE \
  --output_dir \"$TOKENIZER_OUTPUT_DIR\" \
  --use_auth_token True" # Set to False if OSCAR version doesn't require login
run_step "$STEP_NAME" "$COMMAND"

# 2. Continue Pre-training
STEP_NAME="Continue Pre-training"
COMMAND="python run_pretraining.py \
  --base_model_name_or_path \"$BASE_MODEL\" \
  --tokenizer_name_or_path \"$TOKENIZER_OUTPUT_DIR\" \
  --output_dir \"$PRETRAINED_OUTPUT_DIR\" \
  --dataset_name \"$OSCAR_DATASET_NAME\" \
  --dataset_lang \"$OSCAR_DATASET_LANG\" \
  --use_auth_token True \
  --do_train \
  --num_train_epochs $PRETRAINING_EPOCHS \
  --per_device_train_batch_size $PRETRAINING_BATCH_SIZE \
  --learning_rate $PRETRAINING_LR \
  --max_seq_length 512 \
  # --mlm_probability is not used with the custom BART collator
  --mean_span_length 3 \ # Corresponds to lambda=3 in BART paper (default in script)
  --fp16 True \
  --save_steps 10000 \
  --logging_steps 500"
  # Add other TrainingArguments as needed
run_step "$STEP_NAME" "$COMMAND"

# 3. Fine-tune for Summarization
STEP_NAME="Fine-tune for Summarization"
COMMAND="python run_summarization.py \
  --model_name_or_path \"$PRETRAINED_OUTPUT_DIR\" \
  --dataset_name \"$SUMMARIZATION_DATASET_NAME\" \
  # --train_file \"$SUMMARIZATION_TRAIN_FILE\" \ # Uncomment if using local files
  # --validation_file \"$SUMMARIZATION_VALIDATION_FILE\" \ # Uncomment if using local files
  # --test_file \"$SUMMARIZATION_TEST_FILE\" \ # Uncomment if using local files
  --do_train \
  --do_eval \
  --do_predict \
  --output_dir \"$FINETUNED_OUTPUT_DIR\" \
  --num_train_epochs $FINETUNING_EPOCHS \
  --per_device_train_batch_size $FINETUNING_BATCH_SIZE \
  --per_device_eval_batch_size $FINETUNING_BATCH_SIZE \
  --learning_rate $FINETUNING_LR \
  --max_source_length 1024 \
  --max_target_length 128 \
  --predict_with_generate True \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --logging_strategy steps \
  --logging_steps 100 \
  --fp16 True \
  --load_best_model_at_end True \
  --metric_for_best_model rouge2" # Example: optimize for ROUGE-2 F1
  # Add other Seq2SeqTrainingArguments as needed

# Temporarily bypassing run_step for debugging the summarization command
# run_step "$STEP_NAME" "$COMMAND"
echo "--------------------------------------------------"
echo "Running Step: $STEP_NAME (Direct Execution)"
echo "Command: $COMMAND"
echo "--------------------------------------------------"

# --- Debugging: Check FINETUNED_OUTPUT_DIR before execution ---
echo "DEBUG: FINETUNED_OUTPUT_DIR = '$FINETUNED_OUTPUT_DIR'"
if [ -z "$FINETUNED_OUTPUT_DIR" ]; then
  echo "Error: FINETUNED_OUTPUT_DIR is not set. Exiting."
  exit 1
fi
# --- End Debugging ---

# Execute on a single line to rule out line continuation issues
python run_summarization.py --model_name_or_path "$PRETRAINED_OUTPUT_DIR" --dataset_name "$SUMMARIZATION_DATASET_NAME" --do_train --do_eval --do_predict --output_dir "$FINETUNED_OUTPUT_DIR" --num_train_epochs $FINETUNING_EPOCHS --per_device_train_batch_size $FINETUNING_BATCH_SIZE --per_device_eval_batch_size $FINETUNING_BATCH_SIZE --learning_rate $FINETUNING_LR --max_source_length 1024 --max_target_length 128 --predict_with_generate True --evaluation_strategy epoch --save_strategy epoch --logging_strategy steps --logging_steps 100 --fp16 True --load_best_model_at_end True --metric_for_best_model rouge2

if [ $? -ne 0 ]; then
  echo "Error: Step '$STEP_NAME' failed (single-line execution). Exiting."
  exit 1
fi
echo "--------------------------------------------------"
echo "Step '$STEP_NAME' completed successfully."
echo "--------------------------------------------------"


echo "--------------------------------------------------"
echo "Training Pipeline Completed Successfully!"
echo "--------------------------------------------------"

exit 0
