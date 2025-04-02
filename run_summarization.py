import argparse
import logging
import nltk # For ROUGE calculation
import numpy as np
import os
import datasets
from datasets import load_dataset, load_metric
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
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

# --- Argument Parsing ---
# Using HfArgumentParser to easily manage Seq2SeqTrainingArguments and custom args
# Define custom arguments first
parser = argparse.ArgumentParser(description="Fine-tune IndoBART for Summarization.")

# Model/Tokenizer Args
parser.add_argument("--model_name_or_path", type=str, required=True, help="Path to the pre-trained IndoBART model directory")
# Tokenizer path usually same as model path after saving
# parser.add_argument("--tokenizer_name_or_path", type=str, help="Path to the tokenizer directory (if different from model)")

# Data Args
parser.add_argument("--dataset_name", type=str, required=True, help="Hugging Face dataset name or path to local dataset (e.g., 'indosum', 'lipartan6')")
parser.add_argument("--dataset_config_name", type=str, default=None, help="Specific configuration name for the dataset (if applicable)")
parser.add_argument("--document_column", type=str, default="document", help="Column name for the input document/article")
parser.add_argument("--summary_column", type=str, default="summary", help="Column name for the target summary")
parser.add_argument("--train_file", type=str, default=None, help="Path to a custom training file (JSON, CSV)")
parser.add_argument("--validation_file", type=str, default=None, help="Path to a custom validation file (JSON, CSV)")
parser.add_argument("--test_file", type=str, default=None, help="Path to a custom test file (JSON, CSV)")
parser.add_argument("--max_source_length", type=int, default=1024, help="Maximum input sequence length after tokenization.")
parser.add_argument("--max_target_length", type=int, default=128, help="Maximum target sequence length after tokenization.")
parser.add_argument("--preprocessing_num_workers", type=int, default=None, help="Number of processes for data preprocessing")
parser.add_argument("--ignore_pad_token_for_loss", type=bool, default=True, help="Whether to ignore the pad token in loss calculation.")

# Training Args (Handled by Seq2SeqTrainingArguments below, but can add overrides here)
parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")
parser.add_argument("--do_predict", action='store_true', help="Whether to run predictions on the test set.")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory for checkpoints and final model")

# --- Main Script ---
if __name__ == "__main__":
    print("--- Starting Summarization Fine-tuning ---")

    # Parse arguments (basic parsing first)
    args = parser.parse_args()

    # Use HfArgumentParser to parse TrainingArguments alongside custom ones
    # This allows using all standard trainer args like --learning_rate, --num_train_epochs etc.
    hf_parser = HfArgumentParser((Seq2SeqTrainingArguments,))
    # If script args are passed (like --output_dir), they take precedence
    # Need to format args for HfArgumentParser or pass sys.argv directly
    # For simplicity here, we'll manually create TrainingArguments based on parsed args
    # In a full script, using HfArgumentParser with sys.argv is cleaner

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        do_train=args.do_train,
        do_eval=args.do_eval,
        do_predict=args.do_predict,
        # Add common args - users should provide these via command line
        # num_train_epochs=3.0, # Example
        # per_device_train_batch_size=4, # Example
        # per_device_eval_batch_size=4, # Example
        # learning_rate=5e-5, # Example
        # weight_decay=0.01, # Example
        # save_total_limit=3, # Example
        # evaluation_strategy="epoch", # Example
        # logging_strategy="steps", # Example
        # logging_steps=100, # Example
        # predict_with_generate=True, # Crucial for Seq2Seq evaluation
        # fp16=True, # If GPU supports
        # Add generation params like num_beams, length_penalty if needed
        # report_to="wandb", # Example
        # ... other necessary Seq2SeqTrainingArguments ...
        # Need to explicitly pass args from command line or set defaults here
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # 1. Load Pre-trained Model & Tokenizer
    print(f"Loading pre-trained model and tokenizer from: {args.model_name_or_path}")
    try:
        config = AutoConfig.from_pretrained(args.model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
        # Ensure model is loaded for Seq2Seq task
        model = AutoModelForSeq2SeqLM.from_pretrained(
            args.model_name_or_path,
            config=config,
        )
        print("Model and tokenizer loaded.")
    except Exception as e:
        logger.error(f"Error loading model/tokenizer: {e}")
        exit(1)

    # Set prefix for T5-style models if needed (BART usually doesn't require this)
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (AutoTokenizer)): # Check if T5TokenizerFast
         logger.warning("Setting prefix for T5-style model")
         # prefix = "summarize: " # Example prefix

    # 2. Load Dataset
    print(f"Loading dataset: {args.dataset_name}")
    data_files = {}
    if args.train_file: data_files["train"] = args.train_file
    if args.validation_file: data_files["validation"] = args.validation_file
    if args.test_file: data_files["test"] = args.test_file

    try:
        raw_datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            data_files=data_files if data_files else None,
            # cache_dir=model_args.cache_dir, # Add cache dir if needed
        )
        print(f"Dataset loaded. Splits: {raw_datasets.keys()}")
        # print(f"Example train entry: {next(iter(raw_datasets['train']))}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        exit(1)

    # 3. Preprocess Data
    print("Preprocessing dataset...")
    column_names = raw_datasets["train"].column_names # Assuming 'train' split exists

    def preprocess_function(examples):
        inputs = examples[args.document_column]
        targets = examples[args.summary_column]
        # Add prefix if needed
        # inputs = [prefix + inp for inp in inputs]

        model_inputs = tokenizer(inputs, max_length=args.max_source_length, padding="max_length", truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=args.max_target_length, padding="max_length", truncation=True)

        # If we are padding here, replace all tokenizer pad token ids in the labels by -100 when we want to ignore
        # padding in the loss computation.
        if args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    # Apply preprocessing to all splits
    processed_datasets = raw_datasets.map(
        preprocess_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=True, # Enable caching
        desc="Running tokenizer on dataset",
    )

    train_dataset = processed_datasets["train"] if training_args.do_train else None
    eval_dataset = processed_datasets["validation"] if training_args.do_eval else None
    predict_dataset = processed_datasets["test"] if training_args.do_predict else None

    # 4. Data Collator
    label_pad_token_id = -100 if args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None, # Optimize for FP16
    )

    # 5. Metrics (ROUGE)
    try:
        metric = load_metric("rouge")
    except Exception as e:
        logger.error(f"Error loading ROUGE metric: {e}. Make sure 'evaluate' and 'rouge_score' libraries are installed.")
        metric = None

    def compute_metrics(eval_preds):
        if metric is None:
            return {}
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding loss calculation
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # ROUGE expects newline after each sentence
        decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract ROUGE f1 scores
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return {k: round(v, 4) for k, v in result.items()}

    # 6. Initialize Trainer
    print("Initializing Seq2SeqTrainer...")
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # 7. Training, Evaluation, Prediction
    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        # ... (checkpoint detection logic as in pre-training script) ...

    # Training
    if training_args.do_train:
        print("\n--- Starting Fine-tuning Training ---")
        train_result = trainer.train(resume_from_checkpoint=last_checkpoint)
        trainer.save_model()  # Saves the tokenizer too
        # ... (log/save metrics as in pre-training script) ...
        print("\n--- Training Finished ---")

    # Evaluation
    if training_args.do_eval:
        print("\n--- Starting Evaluation ---")
        metrics = trainer.evaluate(max_length=args.max_target_length, num_beams=training_args.generation_num_beams, metric_key_prefix="eval")
        # ... (log/save metrics) ...
        print(f"Evaluation Metrics: {metrics}")
        print("\n--- Evaluation Finished ---")

    # Prediction
    if training_args.do_predict:
        print("\n--- Starting Prediction ---")
        predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict", max_length=args.max_target_length, num_beams=training_args.generation_num_beams)
        # ... (log/save metrics and predictions) ...
        print(f"Prediction Metrics: {predict_results.metrics}")
        # Save predictions if needed
        # ...
        print("\n--- Prediction Finished ---")

    print("\n--- Summarization Fine-tuning Script Outline Complete ---")
    logger.info("Script finished successfully.")

# Example Usage (from command line):
# python run_summarization.py \
#   --model_name_or_path ./indobart_pretrained \
#   --dataset_name indosum \
#   --do_train \
#   --do_eval \
#   --output_dir ./indobart_summarizer \
#   --num_train_epochs 3 \
#   --per_device_train_batch_size 4 \
#   --per_device_eval_batch_size 4 \
#   --learning_rate 3e-5 \
#   --predict_with_generate True \
#   --evaluation_strategy epoch \
#   # Add other Seq2SeqTrainingArguments as needed
