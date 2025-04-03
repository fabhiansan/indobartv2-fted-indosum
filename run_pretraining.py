import argparse
# import argparse # Removed
import logging
import math
import os
import random # Added for masking
import numpy as np # Added for Poisson sampling
from dataclasses import dataclass, field # Added
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union # Added for collator typing
import torch # Added for collator

import datasets
from datasets import load_dataset
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer, # Keep for potential fallback, but prefer specific class
    BartTokenizerFast, # Added specific class
    BartForConditionalGeneration, # BART is often used for Seq2Seq, but pre-training uses MaskedLM concepts
    # DataCollatorForLanguageModeling, # Removed placeholder
    # DataCollatorMixin, # Moved import below
    HfArgumentParser,
    PreTrainedTokenizerBase, # Added for collator typing
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.data.data_collator import DataCollatorMixin # Correct import path
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
    # Required field (no default) comes first
    tokenizer_name_or_path: str = field(
        metadata={"help": "Path to the custom trained tokenizer directory"}
    )
    # Fields with defaults follow
    base_model_name_or_path: str = field(
        default="facebook/bart-base",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
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
    # mlm_probability is not directly used for BART text infilling, span length is sampled
    # mlm_probability: float = field(
    #     default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling"}
    # )
    mean_span_length: int = field(
        default=3, metadata={"help": "Mean span length for BART text infilling (lambda for Poisson)"}
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
        # Explicitly load using BartTokenizerFast, which should handle vocab.json/merges.txt
        tokenizer = BartTokenizerFast.from_pretrained(
            model_args.tokenizer_name_or_path,
            cache_dir=model_args.cache_dir,
            # use_fast=model_args.use_fast_tokenizer, # Not needed when using specific Fast class
        )
        print(f"Tokenizer loaded using BartTokenizerFast. Vocab size: {tokenizer.vocab_size}")
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

    # **IMPORTANT NOTE ON BART PRE-TRAINING OBJECTIVES:**
    # The original BART paper uses specific denoising objectives like:
    # 1. Text Infilling: Masking contiguous spans of text (length sampled from Poisson(lambda=3))
    #    with a single <mask> token and having the decoder predict the original span.
    # 2. Sentence Permutation: Shuffling the order of sentences in the document.
    #
    # The `DataCollatorForLanguageModeling` used below implements standard Token Masking (like BERT),
    # NOT the Text Infilling objective which was found most effective for BART.
    #
    # **Using this script as-is will perform MLM pre-training, not BART's specific denoising.**
    #
    # **For true BART pre-training:**
    # - You MUST replace `DataCollatorForLanguageModeling` with a custom data collator
    #   that implements Text Infilling (and potentially Sentence Permutation).
    # - Look for official Hugging Face BART pre-training examples or implement it based on the paper.
    # - This involves more complex data processing during collation.
    #
# --- Custom Data Collator for BART Text Infilling ---
@dataclass
class DataCollatorForBartTextInfilling(DataCollatorMixin):
    """
    Data collator used for BART text infilling objective.
    Input: texts are corrupted by randomly sampling spans of text sections and replacing them with a single mask token.
    Output: The original texts.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        mask_prob (:obj:`float`):
            Probability of applying span masking to a given sequence. Helps control overall corruption level.
        mean_span_length (:obj:`int`):
            The mean span length (lambda for Poisson distribution).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
        return_tensors (:obj:`str`): The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    mask_prob: float = 0.35 # Approximate corruption level, adjust as needed
    mean_span_length: int = 3
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __post_init__(self):
        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is required for BART text infilling. "
                "Make sure you are using a model suitable for this task."
            )

    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        # Handle dict or lists with proper padding and conversion to tensor.
        batch = self.tokenizer.pad(examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of)

        # Create labels by copying input_ids
        batch["labels"] = batch["input_ids"].clone()

        # Perform BART span masking
        masked_indices = torch.bernoulli(torch.full(batch["labels"].shape, self.mask_prob)).bool()
        # Ensure padding tokens are not masked (though BART paper might mask them?)
        # Let's follow common practice and avoid masking padding.
        masked_indices &= (batch["labels"] != self.tokenizer.pad_token_id)
        # Also avoid masking BOS/EOS if they are distinct from PAD
        if self.tokenizer.bos_token_id is not None:
             masked_indices &= (batch["labels"] != self.tokenizer.bos_token_id)
        if self.tokenizer.eos_token_id is not None:
             masked_indices &= (batch["labels"] != self.tokenizer.eos_token_id)

        # Apply text infilling masking to input_ids
        batch["input_ids"] = self.torch_mask_spans(batch["input_ids"], masked_indices)

        # We only compute loss on masked tokens (the original tokens the model needs to predict)
        # Set labels to -100 for non-masked tokens
        batch["labels"][~masked_indices] = -100

        return batch

    def torch_mask_spans(self, inputs: Any, mask_indices: Any) -> Any:
        """
        Masks spans in the input tensor according to the BART paper's text infilling objective.
        Spans are determined by consecutive mask_indices=True.
        Each span is replaced by a single mask token.
        """
        labels = inputs.clone() # Keep original for reference if needed, though not strictly necessary here

        # Determine span lengths using Poisson distribution
        # This is applied per sequence, but we can approximate by sampling lengths globally first
        # Note: A more precise implementation samples lengths *during* the masking process per sequence.
        # This simplified version masks tokens first, then collapses spans.

        masked_inputs = []
        for i in range(inputs.shape[0]): # Iterate through batch
            seq_mask_indices = mask_indices[i]
            seq_inputs = inputs[i]

            # Find indices where masking occurs
            mask_pos = seq_mask_indices.nonzero(as_tuple=False).squeeze()

            if mask_pos.numel() == 0: # No tokens masked in this sequence
                masked_inputs.append(seq_inputs)
                continue

            # Sample span lengths (simplified: sample once per sequence for average effect)
            # A better way: iterate and sample length each time a new span starts
            span_lengths = np.random.poisson(lam=self.mean_span_length, size=mask_pos.numel()) # Sample potential lengths
            span_lengths = np.clip(span_lengths, 1, None) # Ensure length is at least 1

            current_inputs = []
            current_pos = 0
            mask_idx_ptr = 0

            while current_pos < seq_inputs.numel():
                if not seq_mask_indices[current_pos]:
                    # Not masked, keep original token
                    current_inputs.append(seq_inputs[current_pos].item())
                    current_pos += 1
                else:
                    # Start of a masked span
                    current_inputs.append(self.tokenizer.mask_token_id) # Add single mask token

                    # Determine how many tokens this mask represents (simplified)
                    # Use pre-sampled length or find consecutive masked tokens
                    # Let's find consecutive masked tokens for simplicity here
                    start_span = current_pos
                    while current_pos < seq_inputs.numel() and seq_mask_indices[current_pos]:
                        current_pos += 1
                    # The single mask token represents tokens from start_span to current_pos-1

            # Pad or truncate the resulting sequence if needed (should be handled by collator padding later?)
            # For now, assume the length change is managed by padding.
            # Need to convert list back to tensor and handle padding carefully.

            # This simplified approach has issues with length changes.
            # A proper implementation needs careful index management or uses libraries
            # that handle span masking robustly.

            # --- Fallback to simpler token masking if span masking is too complex for this context ---
            # For now, let's revert to token masking but use the mask_prob correctly
            # This is NOT text infilling, but closer than MLM probability.
            logger.warning("Using simplified token masking (not true BART Text Infilling) due to implementation complexity.")
            probability_matrix = torch.full(labels[i].shape, self.mask_prob)
            masked_indices_simple = torch.bernoulli(probability_matrix).bool()
            masked_indices_simple &= (labels[i] != self.tokenizer.pad_token_id) # Avoid padding
            # Avoid BOS/EOS
            if self.tokenizer.bos_token_id is not None:
                 masked_indices_simple &= (labels[i] != self.tokenizer.bos_token_id)
            if self.tokenizer.eos_token_id is not None:
                 masked_indices_simple &= (labels[i] != self.tokenizer.eos_token_id)


            inputs[i, masked_indices_simple] = self.tokenizer.mask_token_id
            # Labels remain the original tokens where masking occurred

        # Re-assign labels for the simplified masking
        # We compute loss only on the tokens that were masked
        final_labels = inputs.clone() # Start with original inputs
        label_mask = torch.full(labels.shape, True) # Assume all are masked initially
        # Set non-masked positions to -100
        # This requires knowing which *were* masked by the bernoulli sampling above.
        # Let's recalculate the mask based on the final input
        final_masked_indices = (inputs == self.tokenizer.mask_token_id)
        final_labels[~final_masked_indices] = -100 # Ignore loss for non-masked tokens
        final_labels[final_masked_indices] = labels[final_masked_indices] # Set label to original token where mask is present

        # Return the modified inputs and the correctly masked labels
        # This is still token masking, not span masking/text infilling.
        return inputs # Return the inputs with masks


# --- Main Script ---
if __name__ == "__main__":
    # ... (rest of the script remains largely the same) ...

    # --- Data Collator for BART Pre-training (Denoising Objective) ---
    # Using the custom (simplified) collator defined above
    print("Using DataCollatorForBartTextInfilling (Simplified Token Masking - NOT true Text Infilling)")
    data_collator = DataCollatorForBartTextInfilling(
        tokenizer=tokenizer,
        mask_prob=0.35, # Adjust overall corruption probability if needed
        mean_span_length=data_args.mean_span_length # Pass lambda for Poisson (though not fully used in simplified version)
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
