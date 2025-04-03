import argparse
import logging
import math
import os
import datasets
from datasets import load_dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    BartForConditionalGeneration, # BART is often used for Seq2Seq, but pre-training uses MaskedLM concepts
    DataCollatorForLanguageModeling,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# --- Setup Logging ---
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# --- Configuration & Argument Parsing ---
# Define arguments similar to Hugging Face examples for flexibility
# We'll use HfArgumentParser for convenience later if needed, but start simple
parser = argparse.ArgumentParser(description="Continue pre-training BART on Indonesian OSCAR data.")

# Model/Tokenizer Args
parser.add_argument("--base_model_name_or_path", type=str, default="facebook/bart-base", help="Path to pretrained model or model identifier from huggingface.co/models")
parser.add_argument("--tokenizer_name_or_path", type=str, required=True, help="Path to the custom trained tokenizer directory")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory where the final model will be saved")

# Data Args
parser.add_argument("--dataset_name", type=str, default="oscar-corpus/OSCAR-2301", help="Hugging Face dataset name")
parser.add_argument("--dataset_lang", type=str, default="id", help="Language code for the dataset")
parser.add_argument("--use_auth_token", type=bool, default=True, help="Use authentication token for private/gated datasets")
parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="Number of processes for data preprocessing")
parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length for tokenization")
parser.add_argument("--mlm_probability", type=float, default=0.15, help="Ratio of tokens to mask for masked language modeling") # Standard MLM prob

# Training Args (Simplified - use TrainingArguments for full control)
parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
parser.add_argument("--per_device_train_batch_size", type=int, default=24, help="Batch size per GPU/TPU core/CPU for training.")
parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
parser.add_argument("--num_train_epochs", type=float, default=1.0, help="Total number of training epochs to perform.")
# Add more TrainingArguments as needed (e.g., weight_decay, logging_steps, save_steps, gradient_accumulation_steps)

args = parser.parse_args()

# --- Main Script ---
if __name__ == "__main__":
    print("--- Starting Continued Pre-training ---")

    # Set seed before initializing model.
    set_seed(42) # Use a fixed seed for reproducibility

    # 1. Load Custom Tokenizer
    print(f"Loading custom tokenizer from: {args.tokenizer_name_or_path}")
    try:
        # Use AutoTokenizer to load from the saved directory
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name_or_path)
        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        exit(1)

    # 2. Load Base Model Config & Model
    print(f"Loading base model config and weights: {args.base_model_name_or_path}")
    try:
        # Load config first
        config = AutoConfig.from_pretrained(args.base_model_name_or_path)
        # Update config with tokenizer specifics if needed (e.g., vocab size, special tokens)
        # config.vocab_size = tokenizer.vocab_size # Often handled by resize_token_embeddings

        # Load the model. BART is conditional generation, but pre-training often uses MLM head.
        # Let's try loading as BartForConditionalGeneration first, as that's its primary architecture.
        # If MLM-style pre-training is strictly needed, AutoModelForMaskedLM might be used,
        # but adapting BART's Seq2Seq pre-training (denoising) is more standard.
        # For simplicity here, we load the base architecture. The Trainer handles loss internally.
        model = BartForConditionalGeneration.from_pretrained(
            args.base_model_name_or_path,
            config=config,
        )
        print("Base model loaded.")
    except Exception as e:
        logger.error(f"Error loading base model: {e}")
        exit(1)

    # 3. Resize Token Embeddings
    print(f"Resizing token embeddings to match tokenizer vocab size: {tokenizer.vocab_size}")
    model.resize_token_embeddings(len(tokenizer))
    # Check if resizing worked (optional)
    if model.get_input_embeddings().weight.shape[0] != len(tokenizer):
         logger.warning("Embedding size mismatch after resizing. Check model/tokenizer compatibility.")


    # 4. Load and Prepare Dataset
    print(f"Loading dataset: {args.dataset_name}, Language: {args.dataset_lang}")
    try:
        # Load dataset (consider streaming for large datasets)
        # For pre-training, we usually just need the 'train' split
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_lang,
            split="train",
            streaming=False, # Set to True for very large datasets, adjust preprocessing
            use_auth_token=args.use_auth_token
        )
        print(f"Dataset loaded. Example entry: {next(iter(raw_datasets))}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        exit(1)

    # Preprocessing function
    def tokenize_function(examples):
        # Assuming text is in 'text' or 'content' field
        text_field = 'text' if 'text' in examples else 'content'
        # Tokenize the texts
        return tokenizer(examples[text_field], return_special_tokens_mask=True) # MLM needs special tokens mask

    print("Tokenizing dataset...")
    # Note: For large datasets, map might be slow without streaming/batching
    tokenized_datasets = raw_datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=raw_datasets.column_names, # Remove original text columns
        load_from_cache_file=True, # Enable caching
        desc="Running tokenizer on dataset",
    )

    # --- Data Collator for BART Pre-training (Denoising Objective) ---
    # BART's pre-training is different from standard MLM (like BERT).
    # It involves corrupting the input (e.g., masking spans, permuting sentences)
    # and reconstructing the original text.
    # `DataCollatorForLanguageModeling` is for standard MLM.
    # Implementing BART's specific denoising requires a custom collator or
    # adapting existing Seq2Seq collators if fine-tuning pre-training scripts.

    # **Placeholder:** Using standard MLM collator for simplicity in this template.
    # **Actual Implementation:** Would need a custom collator for BART's text infilling.
    # See Hugging Face documentation/examples for BART pre-training specifics.
    print("Using standard DataCollatorForLanguageModeling (Placeholder - BART needs specific denoising collator)")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True, # Enable masking
        mlm_probability=args.mlm_probability
    )

    # 5. Configure Training Arguments
    # Use TrainingArguments for full control
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True, # Be careful with this
        do_train=args.do_train,
        # Add other necessary arguments:
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        learning_rate=args.learning_rate,
        save_steps=10_000, # Example: Save checkpoint every 10k steps
        save_total_limit=2, # Example: Keep only the last 2 checkpoints
        logging_steps=500, # Example: Log every 500 steps
        fp16=True, # Enable mixed precision if GPU supports it
        # Add gradient_accumulation_steps, weight_decay, warmup_steps etc.
        # report_to="wandb", # Example: Integrate with Weights & Biases
    )
    logger.info(f"Training/evaluation parameters {training_args}")


    # 6. Initialize Trainer
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets, # Pass the processed dataset
        # eval_dataset=tokenized_datasets["validation"] if "validation" in tokenized_datasets else None, # Add validation if available
        tokenizer=tokenizer,
        data_collator=data_collator, # Use the appropriate collator
    )

    # 7. Run Training (if requested)
    if args.do_train:
        print("\n--- Starting Training ---")
        # Check for last checkpoint
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()  # Saves the tokenizer too

        metrics = train_result.metrics
        # Log metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        print("\n--- Training Finished ---")
    else:
        print("\nSkipping training as --do_train was not specified.")
        print("This script sets up the model, tokenizer, and data pipeline.")
        print("Run with --do_train to execute the pre-training.")

    print("\n--- Continued Pre-training Script Outline Complete ---")
    logger.info("Script finished successfully.")

# Example Usage (from command line):
# python run_pretraining.py \
#   --tokenizer_name_or_path ./indonesian_tokenizer \
#   --output_dir ./indobart_pretrained \
#   --do_train \
#   --num_train_epochs 1 \
#   --per_device_train_batch_size 8 \
#   --learning_rate 5e-5 \
#   # Add other TrainingArguments as needed
