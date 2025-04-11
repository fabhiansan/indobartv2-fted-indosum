# finetune_config.py

import torch
from transformers import TrainingArguments, Seq2SeqTrainingArguments

# --- Model Configuration ---
BASE_MODEL_NAME = "gaduhhartawan/indobart-base-v2"
# Directory where the fine-tuned model and tokenizer will be saved
OUTPUT_DIR = "./indobart-finetuned-summarization"

# --- Dataset Configuration ---
# Note: XSUM is English, the others are Indonesian.
# The script will handle combining them, but be mindful of potential language mixing effects.
DATASET_NAMES = {
    "xsum": "xsum", # Standard English summarization dataset
    "indosum": "SEACrowd/indosum", # Indonesian summarization dataset
    "liputan6": "fajrikoto/id_liputan6" # Indonesian news summarization dataset
}
# Column names expected in the datasets (adjust if necessary after inspection)
# These might need verification by loading a sample of each dataset.
DATASET_COLUMNS = {
    "xsum": {"document": "document", "summary": "summary"},
    "indosum": {"document": "text", "summary": "summary"}, # Assuming based on common patterns
    "liputan6": {"document": "clean_article", "summary": "clean_summary"} # Assuming based on common patterns
}
# Use a subset for faster testing/debugging (set to None to use full datasets)
MAX_TRAIN_SAMPLES = 1000 # Example: Use only 1000 training samples per dataset
MAX_EVAL_SAMPLES = 200   # Example: Use only 200 evaluation samples per dataset

# --- Tokenizer Configuration ---
# Max length for input documents (articles)
MAX_INPUT_LENGTH = 1024
# Max length for output summaries
MAX_TARGET_LENGTH = 128
# Prefix for summarization tasks (optional, but can sometimes help)
SUMMARIZATION_PREFIX = "summarize: "

# --- Training Configuration ---
NUM_TRAIN_EPOCHS = 3
PER_DEVICE_TRAIN_BATCH_SIZE = 4 # Adjust based on GPU memory
PER_DEVICE_EVAL_BATCH_SIZE = 8  # Adjust based on GPU memory
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 500
FP16 = torch.cuda.is_available() # Use mixed precision if CUDA is available

# --- Logging and Saving ---
LOGGING_STRATEGY = "epoch"
EVALUATION_STRATEGY = "epoch"
SAVE_STRATEGY = "epoch"
SAVE_TOTAL_LIMIT = 2 # Keep only the best and the latest checkpoint
LOAD_BEST_MODEL_AT_END = True
METRIC_FOR_BEST_MODEL = "loss" # Use validation loss to determine the best model
GREATER_IS_BETTER = False # For loss, lower is better

# --- Seq2Seq Training Arguments ---
# DO NOT EDIT training_args directly here. It's constructed in the main script.
# This is just a placeholder showing the parameters used.
# training_args = Seq2SeqTrainingArguments(
#     output_dir=OUTPUT_DIR,
#     num_train_epochs=NUM_TRAIN_EPOCHS,
#     per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
#     per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
#     learning_rate=LEARNING_RATE,
#     weight_decay=WEIGHT_DECAY,
#     warmup_steps=WARMUP_STEPS,
#     fp16=FP16,
#     evaluation_strategy=EVALUATION_STRATEGY,
#     save_strategy=SAVE_STRATEGY,
#     logging_strategy=LOGGING_STRATEGY,
#     save_total_limit=SAVE_TOTAL_LIMIT,
#     load_best_model_at_end=LOAD_BEST_MODEL_AT_END,
#     metric_for_best_model=METRIC_FOR_BEST_MODEL,
#     greater_is_better=GREATER_IS_BETTER,
#     predict_with_generate=True, # Important for Seq2Seq tasks
#     report_to="tensorboard", # Or "wandb", "none"
#     push_to_hub=False, # Set to True to push to Hugging Face Hub
# )