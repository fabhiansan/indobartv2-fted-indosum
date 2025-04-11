# run_finetuning.py

import logging
import os
import sys
import nltk # Needed for ROUGE score calculation (tokenization)
import numpy as np
from datasets import load_metric
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
import finetune_config as cfg
from data_utils import load_and_prepare_datasets

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Download nltk punkt tokenizer if not already present
try:
    nltk.data.find("tokenizers/punkt")
except nltk.downloader.DownloadError:
    logger.info("Downloading nltk punkt tokenizer...")
    nltk.download("punkt", quiet=True)

# Load ROUGE metric
try:
    # Note: Using load_metric from datasets for compatibility,
    # but 'evaluate' library is the newer standard.
    # If 'load_metric' fails, try 'evaluate.load("rouge")'
    rouge_metric = load_metric("rouge")
except Exception as e:
     logger.error(f"Failed to load ROUGE metric using datasets.load_metric: {e}. Trying evaluate.load...", exc_info=False)
     try:
         import evaluate
         rouge_metric = evaluate.load("rouge")
         logger.info("Successfully loaded ROUGE metric using 'evaluate' library.")
     except ImportError:
          logger.error("Failed to load ROUGE metric. 'evaluate' library not found. Please install it: pip install evaluate rouge_score")
          sys.exit(1)
     except Exception as e2:
          logger.error(f"Failed to load ROUGE metric using 'evaluate' library: {e2}. Please ensure 'rouge_score' is also installed: pip install evaluate rouge_score", exc_info=True)
          sys.exit(1)


def compute_metrics(eval_pred):
    """Computes ROUGE scores for evaluation."""
    predictions, labels = eval_pred
    # Decode generated summaries (replace -100 labels used for padding)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Simple text cleaning for ROUGE
    decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
    decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

    # Compute ROUGE scores
    result = rouge_metric.compute(
        predictions=decoded_preds, references=decoded_labels, use_stemmer=True
    )
    # Extract ROUGE f-measures if using datasets.load_metric format
    if hasattr(next(iter(result.values())), 'mid'): # Check if it's the old format
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
    # Otherwise, assume evaluate.load format (already floats) - multiply by 100
    else:
         result = {key: value * 100 for key, value in result.items()}


    # Add mean generated length
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


def main():
    logger.info("--- Starting Fine-tuning Script ---")
    set_seed(42) # Set seed for reproducibility

    # 1. Load Tokenizer and Model
    logger.info(f"Loading tokenizer: {cfg.BASE_MODEL_NAME}")
    # Declare tokenizer globally within main for compute_metrics access
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL_NAME)


    logger.info(f"Loading model: {cfg.BASE_MODEL_NAME}")
    model = AutoModelForSeq2SeqLM.from_pretrained(cfg.BASE_MODEL_NAME)

    # 2. Load and Prepare Datasets
    logger.info("Loading and preparing datasets...")
    try:
        tokenized_datasets = load_and_prepare_datasets(tokenizer)
        if not tokenized_datasets:
             logger.error("Dataset preparation returned empty. Exiting.")
             return
        logger.info("Datasets prepared successfully.")
        logger.info(f"Available splits: {list(tokenized_datasets.keys())}")
    except Exception as e:
        logger.error(f"Failed during dataset preparation: {e}", exc_info=True)
        return

    # Ensure required splits exist
    train_dataset = tokenized_datasets.get("train")
    eval_dataset = tokenized_datasets.get("validation")
    if not train_dataset:
        logger.error("Training dataset not found after processing. Exiting.")
        return
    if not eval_dataset:
        logger.warning("Validation dataset not found after processing. Evaluation during training will be skipped.")
        # Optionally disable evaluation if eval_dataset is None
        # cfg.EVALUATION_STRATEGY = "no"


    # 3. Define Training Arguments
    logger.info("Defining Training Arguments...")
    training_args = Seq2SeqTrainingArguments(
        output_dir=cfg.OUTPUT_DIR,
        num_train_epochs=cfg.NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=cfg.PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=cfg.PER_DEVICE_EVAL_BATCH_SIZE,
        learning_rate=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
        warmup_steps=cfg.WARMUP_STEPS,
        fp16=cfg.FP16,
        evaluation_strategy=cfg.EVALUATION_STRATEGY if eval_dataset else "no", # Only eval if dataset exists
        save_strategy=cfg.SAVE_STRATEGY,
        logging_strategy=cfg.LOGGING_STRATEGY,
        save_total_limit=cfg.SAVE_TOTAL_LIMIT,
        load_best_model_at_end=cfg.LOAD_BEST_MODEL_AT_END if eval_dataset else False, # Only load best if eval happens
        metric_for_best_model=cfg.METRIC_FOR_BEST_MODEL if eval_dataset else None,
        greater_is_better=cfg.GREATER_IS_BETTER,
        predict_with_generate=True, # Generate summaries during evaluation
        generation_max_length=cfg.MAX_TARGET_LENGTH, # Control generation length
        report_to="tensorboard", # Logs to ./runs by default
        push_to_hub=False,
        seed=42, # Ensure reproducibility in trainer too
        # Add generation config if needed, e.g., num_beams
        # generation_num_beams=4,
    )
    logger.info(f"Training Arguments: {training_args.to_dict()}")

    # 4. Define Data Collator
    logger.info("Defining Data Collator...")
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100, # Standard practice for ignoring padding in loss calculation
        pad_to_multiple_of=8 if training_args.fp16 else None # Optimize for FP16
    )

    # 5. Initialize Trainer
    logger.info("Initializing Seq2SeqTrainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset, # Will be None if validation set wasn't found
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset else None # Only compute metrics if eval happens
    )

    # 6. Train
    logger.info("--- Starting Training ---")
    try:
        train_result = trainer.train()
        logger.info("--- Training Finished ---")

        # Save final model, tokenizer, and training stats
        logger.info("Saving final model and tokenizer...")
        trainer.save_model() # Saves the tokenizer too
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()
        logger.info(f"Model and tokenizer saved to {cfg.OUTPUT_DIR}")

        # 7. Evaluate (if test set exists and evaluation happened)
        if "test" in tokenized_datasets and eval_dataset:
            logger.info("--- Starting Final Evaluation on Test Set ---")
            test_dataset = tokenized_datasets["test"]
            test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
            logger.info(f"Test Set Results: {test_results}")
            trainer.log_metrics("test", test_results)
            trainer.save_metrics("test", test_results)
        elif "test" in tokenized_datasets:
             logger.warning("Test set exists but evaluation was skipped (no validation set). Cannot run final evaluation.")
        else:
            logger.info("No test set found. Skipping final evaluation.")


    except Exception as e:
        logger.error(f"An error occurred during training or evaluation: {e}", exc_info=True)

    logger.info("--- Fine-tuning Script Finished ---")

if __name__ == "__main__":
    # Ensure necessary libraries are installed
    try:
        import datasets
        import transformers
        import rouge_score
        import evaluate
        import nltk
    except ImportError as e:
        logger.error(f"Missing required library: {e}. Please install dependencies: pip install datasets transformers evaluate rouge_score nltk torch sentencepiece") # Added sentencepiece
        sys.exit(1)

    main()