import argparse
import logging
import math
import os
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    BartTokenizerFast,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """
    model_output_dir: str = field(
        metadata={"help": "Path to save the final model."}
    )
    tokenizer_path: str = field(
        metadata={"help": "Path to the pretrained tokenizer directory."}
    )
    config_name: Optional[str] = field(
        default="facebook/bart-base", metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"}
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
    cleaned_text_file: str = field(
        metadata={"help": "Path to the cleaned input text file (output of preprocess_data.py)."}
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    text_column_name: Optional[str] = field(
        default="text",
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )


@dataclass
class DataCollatorForBartDenoising:
    """
    Data collator used for BART denoising task with text infilling.
    Reference: https://github.com/huggingface/transformers/blob/main/examples/pytorch/language-modeling/run_mlm.py
    Slightly adapted for text infilling based on BART paper.
    """
    tokenizer: BartTokenizerFast
    mask_prob: float = 0.30  # Probability of masking tokens (BART uses 0.3)
    mean_span_length: float = 3.0 # Mean length of spans to mask (BART uses Poisson(lambda=3))

    def __post_init__(self):
        if self.tokenizer.mask_token is None or self.tokenizer.pad_token is None:
            raise ValueError("This collator requires a tokenizer with `mask_token` and `pad_token` symbols.")
        self.mask_token_id = self.tokenizer.mask_token_id
        self.pad_token_id = self.tokenizer.pad_token_id
        self.decoder_start_token_id = self.tokenizer.bos_token_id # BART uses <s> as decoder start
        # Precompute Poisson probabilities for span lengths
        self.poisson_distribution = [
            (self.mean_span_length**k / math.factorial(k)) * math.exp(-self.mean_span_length)
            for k in range(1, 40) # Calculate for lengths 1 to 39
        ]
        self.poisson_distribution = np.array(self.poisson_distribution)
        self.poisson_distribution /= self.poisson_distribution.sum() # Normalize

    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Examples are list of dicts with keys like 'input_ids', 'attention_mask'
        # We need to corrupt the input_ids for the encoder and create labels for the decoder

        input_ids = [example["input_ids"] for example in examples]
        batch_size = len(input_ids)

        # Convert list of lists to tensor for easier manipulation if needed, handle padding later
        # Keep as lists for now to handle variable lengths during corruption

        corrupted_input_ids = []
        labels = []

        for ids in input_ids:
            # Clone the original ids to corrupt
            corrupted = list(ids)
            original_len = len(ids) # Length before potential padding

            n_masks_to_add = int(round(original_len * self.mask_prob))
            n_masked = 0

            masked_indices = np.zeros(original_len, dtype=bool) # Track which indices are already part of a mask span

            while n_masked < n_masks_to_add:
                # Sample span length from Poisson distribution
                span_len = np.random.choice(np.arange(1, len(self.poisson_distribution) + 1), p=self.poisson_distribution)
                # Find a random start index for the span
                anchor = np.random.randint(0, original_len - span_len + 1)

                # Ensure we don't overlap with already masked spans or mask special tokens
                # This simple check might not be perfect for complex overlaps
                is_valid_span = True
                for i in range(anchor, anchor + span_len):
                     # Avoid masking special tokens (assuming they are at ends or specific IDs)
                    if corrupted[i] in self.tokenizer.all_special_ids or masked_indices[i]:
                        is_valid_span = False
                        break
                if not is_valid_span:
                    continue # Try another anchor/span

                # Mark indices as masked for this span
                for i in range(anchor, anchor + span_len):
                    masked_indices[i] = True

                n_masked += span_len

            # Now apply the masking based on masked_indices
            # Replace consecutive masked spans with a single mask token
            corrupted_final = []
            i = 0
            while i < original_len:
                if masked_indices[i]:
                    corrupted_final.append(self.mask_token_id)
                    # Skip the rest of the masked span
                    while i < original_len and masked_indices[i]:
                        i += 1
                else:
                    corrupted_final.append(corrupted[i])
                    i += 1

            # Labels are the original tokens
            # Pad labels with -100 where input is not masked (or originally padding)
            # Note: The Trainer handles label padding automatically if labels match input length before corruption
            # For denoising, labels should match the *original* uncorrupted sequence length
            label = list(ids) # Start with original ids

            corrupted_input_ids.append(torch.tensor(corrupted_final, dtype=torch.long))
            labels.append(torch.tensor(label, dtype=torch.long))


        # Pad the corrupted inputs and the original labels
        padded_inputs = self.tokenizer.pad(
            {"input_ids": corrupted_input_ids},
            padding="longest",
            return_tensors="pt",
        )
        padded_labels = self.tokenizer.pad(
             {"input_ids": labels}, # Pad the original sequences
             padding="longest",
             return_tensors="pt",
        ).input_ids # Get only the padded ids

        # Set padding token labels to -100
        padded_labels[padded_labels == self.pad_token_id] = -100

        # Decoder input ids are shifted right for BART
        # Handled internally by BartForConditionalGeneration if decoder_input_ids are not provided
        # We just need to provide input_ids (corrupted) and labels (original)

        return {
            "input_ids": padded_inputs["input_ids"],
            "attention_mask": padded_inputs["attention_mask"],
            "labels": padded_labels,
        }


def main():
    # --- Parse Arguments ---
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # --- Setup ---
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.setLevel(logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN)
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    # --- Check for Checkpoints ---
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # --- Load Tokenizer ---
    logger.info(f"Loading tokenizer from {model_args.tokenizer_path}")
    tokenizer = BartTokenizerFast.from_pretrained(
        model_args.tokenizer_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )

    # --- Load Dataset ---
    logger.info(f"Loading text dataset from {data_args.cleaned_text_file}")
    # Load as text dataset
    datasets = load_dataset("text", data_files=data_args.cleaned_text_file, cache_dir=model_args.cache_dir)
    # Assuming the dataset has a 'train' split after loading from text file
    if "train" not in datasets:
         raise ValueError(f"Dataset loaded from {data_args.cleaned_text_file} does not contain a 'train' split.")
    column_names = datasets["train"].column_names
    text_column_name = data_args.text_column_name if data_args.text_column_name in column_names else column_names[0]


    # --- Tokenize Dataset ---
    logger.info("Tokenizing dataset...")
    max_seq_length = data_args.max_seq_length

    def tokenize_function(examples):
        return tokenizer(
            examples[text_column_name],
            padding=False, # Collator will handle padding
            truncation=True,
            max_length=max_seq_length,
            # We could return special tokens mask etc. if needed by collator, but basic BART needs input_ids
        )

    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names, # Remove original text column
        load_from_cache_file=not data_args.overwrite_cache,
        desc="Running tokenizer on dataset",
    )
    train_dataset = tokenized_datasets["train"]
    logger.info(f"Number of training examples: {len(train_dataset)}")
    # Example tokenized entry
    logger.info(f"Example tokenized input_ids: {train_dataset[0]['input_ids'][:50]}...")


    # --- Load Model Config & Instantiate Model ---
    logger.info("Loading model configuration and initializing model from scratch...")
    config = BartConfig.from_pretrained(
        model_args.config_name,
        cache_dir=model_args.cache_dir,
    )
    # !!! Crucial: Set vocab size to match tokenizer !!!
    logger.warning(f"Setting model config vocab size to tokenizer vocab size: {tokenizer.vocab_size}")
    config.vocab_size = tokenizer.vocab_size
    # Ensure pad, bos, eos token IDs are consistent (usually handled by tokenizer loading)
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    # BART uses bos_token_id as decoder_start_token_id
    config.decoder_start_token_id = tokenizer.bos_token_id

    model = BartForConditionalGeneration(config=config)
    model.resize_token_embeddings(len(tokenizer)) # Ensure embedding size matches tokenizer with potentially added special tokens

    # --- Initialize Data Collator ---
    logger.info("Initializing data collator...")
    data_collator = DataCollatorForBartDenoising(
        tokenizer=tokenizer,
        mask_prob=0.30, # Standard BART value
        mean_span_length=3.0 # Standard BART value
    )

    # --- Initialize Trainer ---
    logger.info("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        # eval_dataset=eval_dataset if training_args.do_eval else None, # No eval dataset specified for pretraining here
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # --- Training ---
    if training_args.do_train:
        logger.info("*** Starting Training ***")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(model_args.model_output_dir)  # Saves the tokenizer too

        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        logger.info(f"Training finished. Model saved to {model_args.model_output_dir}")

    # --- Evaluation (Optional) ---
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     # Add evaluation logic here if needed

    logger.info("Script finished.")


if __name__ == "__main__":
    main()
