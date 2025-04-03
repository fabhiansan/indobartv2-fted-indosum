import argparse
# import argparse # Removed
import logging
import math
import os
from dataclasses import dataclass, field # Added
from typing import Optional # Added
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

# --- Define Argument Classes (for HfArgumentParser) ---
# Use dataclasses for better organization and type hints
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    base_model_name_or_path: str = field(
        default="facebook/bart-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    tokenizer_name_or_path: str = field(
        metadata={"help": "Path to the custom trained tokenizer directory"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    dataset_name: Optional[str] = field(
        default="oscar-corpus/OSCAR-2301", metadata={"help": "Hugging Face dataset name"}
    )
    dataset_lang: Optional[str] = field(
        default="id", metadata={"help": "Language code for the dataset"}
    )
    use_auth_token: bool = field(
        default=True, metadata={"help": "Use authentication token for private/gated datasets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "Number of processes for data preprocessing"}
    )
    max_seq_length: Optional[int] = field(
        default=512, metadata={"help": "Maximum sequence length for tokenization"}
    )
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling"}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


# --- Main Script ---
if __name__ == "__main__":
    print("--- Starting Continued Pre-training ---")

    # --- Parse Arguments using HfArgumentParser ---
    # This automatically handles TrainingArguments along with custom ones
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    # Example: If you need JSON input:
    # model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    # Example: Standard command-line parsing:
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging based on training_args
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")

    print("--- Starting Continued Pre-training ---")

    # Set seed before initializing model.
    set_seed(training_args.seed) # Use seed from training_args

    # 1. Load Custom Tokenizer
    print(f"Loading custom tokenizer from: {model_args.tokenizer_name_or_path}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            use_fast=model_args.use_fast_tokenizer,
        )
        print(f"Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        logger.error(f"Error loading tokenizer: {e}")
        exit(1)

    # 2. Load Base Model Config & Model
    print(f"Loading base model config and weights: {model_args.base_model_name_or_path}")
    try:
        config = AutoConfig.from_pretrained(
            model_args.config_name if model_args.config_name else model_args.base_model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        # Update config with tokenizer specifics if needed (e.g., vocab size, special tokens)
        # config.vocab_size = tokenizer.vocab_size # Often handled by resize_token_embeddings

        # Load the model. BART is conditional generation, but pre-training often uses MLM head.
        # Let's try loading as BartForConditionalGeneration first, as that's its primary architecture.
        # If MLM-style pre-training is strictly needed, AutoModelForMaskedLM might be used,
        # but adapting BART's Seq2Seq pre-training (denoising) is more standard.
        # For simplicity here, we load the base architecture. The Trainer handles loss internally.
        model = BartForConditionalGeneration.from_pretrained(
            model_args.base_model_name_or_path,
            from_tf=bool(".ckpt" in model_args.base_model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
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
    print(f"Loading dataset: {data_args.dataset_name}, Language: {data_args.dataset_lang}")
    try:
        # Load dataset (consider streaming for large datasets)
        # For pre-training, we usually just need the 'train' split
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_lang,
            split="train",
            streaming=False, # Set to True for very large datasets, adjust preprocessing
            use_auth_token=data_args.use_auth_token,
            cache_dir=model_args.cache_dir,
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
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=raw_datasets.column_names, # Remove original text columns
        load_from_cache_file=not data_args.overwrite_cache, # Control caching
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
        mlm_probability=data_args.mlm_probability
    )

    # 5. Configure Training Arguments is already done by HfArgumentParser
    # We now use training_args directly, which is an instance of TrainingArguments

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
    if training_args.do_train: # Use training_args.do_train
        print("\n--- Starting Training ---")
        # Check for last checkpoint
        last_checkpoint = None
        # Detecting last checkpoint logic from Hugging Face examples
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                 # Check if output directory exists and is not empty, and overwrite_output_dir is False
                 # This check might need refinement based on exact HF script logic
                 # For now, assume HfArgumentParser handles this or raise error if needed
                 logger.info(f"Output directory ({training_args.output_dir}) already exists and is not empty. Not overwriting.")
                 # Or potentially raise error if overwrite_output_dir is False and dir is not empty
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        # Determine checkpoint to resume from
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too

        metrics = train_result.metrics
        # Add max_train_samples logic if needed for metrics
        # max_train_samples = (
        #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        # )
        # metrics["train_samples"] = min(max_train_samples, len(train_dataset))

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
# Note: Pass TrainingArguments like --fp16, --save_steps directly now
# python run_pretraining.py \
#   --tokenizer_name_or_path ./indonesian_tokenizer \
#   --output_dir ./indobart_pretrained \
#   --do_train \
#   --num_train_epochs 1 \
#   --per_device_train_batch_size 8 \
#   --learning_rate 5e-5 \
#   # Add other TrainingArguments as needed
