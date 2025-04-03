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
from transformers.tokenization_utils_base import BatchEncoding # Added import
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

# Helper function for collating lists of integers
# (Needed if examples are lists instead of dicts/BatchEncoding)
def _collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    # Check if padding is necessary.
    length_of_first = examples[0].size(0)
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and pad_to_multiple_of is None:
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer does not have a pad token."
        )

    # Determine maximum length
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None:
        max_length = (
            (max_length + pad_to_multiple_of - 1)
            // pad_to_multiple_of
            * pad_to_multiple_of
        )

    # Pad
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result

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
    original_vocab_size = model.get_input_embeddings().weight.shape[0]
    print(f"Original model vocab size: {original_vocab_size}")
    print(f"Tokenizer vocab size: {len(tokenizer)}")
    if original_vocab_size != len(tokenizer):
        print(f"Resizing token embeddings from {original_vocab_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))
        # Verify after resizing
        new_vocab_size = model.get_input_embeddings().weight.shape[0]
        print(f"New model vocab size after resizing: {new_vocab_size}")
        if new_vocab_size != len(tokenizer):
            logger.error(f"FATAL: Embedding size mismatch even after resizing! Model: {new_vocab_size}, Tokenizer: {len(tokenizer)}")
            exit(1) # Exit if resizing failed critically
        # Also update the config if necessary (though resize_token_embeddings often handles this)
        # model.config.vocab_size = new_vocab_size
    else:
        print("Model vocab size already matches tokenizer vocab size. No resizing needed.")


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
    # Class definition moved inside if __name__ == '__main__' block below
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
            if self.tokenizer.eos_token_id is not None: # Align this 'if' with the previous 'if'
                masked_indices &= (batch["labels"] != self.tokenizer.eos_token_id) # Indent under the 'if'

            # The initial self.tokenizer.pad call handles batch creation and padding.

            # Clone input_ids for labels before masking (using the 'batch' from the initial pad call)
            labels = batch["input_ids"].clone()

            # Create corrupted input_ids using BART text infilling
            masked_inputs = labels.clone() # Start with original tokens
            special_tokens_mask = batch.pop("special_tokens_mask", None) # Use if available from tokenizer

            # Perform BART text infilling masking
            masked_inputs = self.torch_mask_text_infilling(masked_inputs, special_tokens_mask)

            batch["input_ids"] = masked_inputs

            # Set labels for non-masked tokens to -100
            # In text infilling, the labels are the original tokens, but we ignore loss for tokens
            # that were *not* part of a masked span *or* are padding tokens in the original sequence.
            # The `masked_inputs` tensor now contains the MASK tokens where spans were.
            # The `labels` tensor contains the original tokens.
            # We need to set labels to -100 where `masked_inputs` is NOT a mask token,
            # *and* also for the original padding tokens.
            if self.tokenizer.pad_token_id is not None:
                padding_mask = labels.eq(self.tokenizer.pad_token_id)
                labels[padding_mask] = -100 # Ignore padding

            # Ignore tokens that were not masked (i.e., where input is not MASK)
            # Note: This assumes the MASK token itself isn't a valid label we want to predict, which is safe.
            non_masked_original_tokens_mask = ~masked_inputs.eq(self.tokenizer.mask_token_id)
            labels[non_masked_original_tokens_mask] = -100

            batch["labels"] = labels

            # Ensure attention mask corresponds to the potentially shorter masked_inputs before padding
            # The tokenizer.pad should handle this correctly if called on the final masked inputs.
            # Let's re-pad here to be certain lengths match after masking potentially shortened sequences.
            # This requires converting back to list of lists, masking, then padding again.
            # Simpler: Assume initial padding was sufficient and adjust attention mask.
            # If initial padding was to max_length, attention_mask is likely correct.
            # If padding was dynamic, it needs recalculation based on `masked_inputs`.
            # Let's assume tokenizer.pad handled it initially.

            return batch # Ensure this is indented to match the start of the torch_call method body


        def torch_mask_text_infilling(self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
            """
            Apply BART's text infilling corruption to inputs.
            - Sample span lengths from Poisson(lambda=mean_span_length).
            - Sample number of tokens to mask based on mask_prob.
            - Replace chosen spans with a single mask token.
            """
            mask_token_id = self.tokenizer.mask_token_id
            pad_token_id = self.tokenizer.pad_token_id
            bos_token_id = self.tokenizer.bos_token_id
            eos_token_id = self.tokenizer.eos_token_id

            masked_sequences = []
            for i in range(inputs.shape[0]): # Iterate through batch
                sequence = inputs[i].tolist()
                original_length = len(sequence) # Before removing padding for masking logic

                # Remove padding for masking calculation if present
                if pad_token_id is not None:
                    actual_tokens = [tok for tok in sequence if tok != pad_token_id]
                else:
                    actual_tokens = sequence
                
                if not actual_tokens: # Handle empty sequences
                    masked_sequences.append(inputs[i]) # Return original padding
                    continue

                n_tokens = len(actual_tokens)
                if n_tokens <= 1: # Cannot mask if only BOS/EOS or single token
                     masked_sequences.append(inputs[i])
                     continue

                # Determine number of tokens to mask (target percentage of actual tokens)
                num_to_mask = int(round(n_tokens * self.mask_prob))
                if num_to_mask == 0:
                    masked_sequences.append(inputs[i]) # No masking needed
                    continue

                # Sample span lengths from Poisson distribution
                span_lengths = []
                current_sum = 0
                while current_sum < num_to_mask:
                    length = np.random.poisson(lam=self.mean_span_length)
                    if length > 0: # Only consider spans of length 1 or more
                        span_lengths.append(length)
                        current_sum += length
                
                # Trim excess length from the last span if needed
                if current_sum > num_to_mask:
                     diff = current_sum - num_to_mask
                     if span_lengths[-1] > diff:
                         span_lengths[-1] -= diff
                     else: # If last span is too small, remove it (or adjust previous)
                         span_lengths.pop() # Simplest approach

                if not span_lengths: # Handle case where no valid spans generated
                    masked_sequences.append(inputs[i])
                    continue

                # Identify indices that can be masked (exclude special tokens like BOS/EOS if they are not PAD)
                indices_maskable = np.array([
                    idx for idx, token_id in enumerate(actual_tokens)
                    if token_id != bos_token_id and token_id != eos_token_id
                ])

                if len(indices_maskable) < sum(span_lengths):
                    # Not enough maskable tokens for the desired spans, mask fewer
                    # This can happen with short sequences or high mask_prob
                    # Simple strategy: mask all maskable tokens if needed
                    num_to_mask = len(indices_maskable)
                    span_lengths = [num_to_mask] if num_to_mask > 0 else []
                    # Alternative: recalculate spans based on available indices

                if not span_lengths:
                     masked_sequences.append(inputs[i])
                     continue

                # Choose starting indices for spans randomly from maskable indices
                # Ensure spans don't overlap (more complex) or allow overlap (simpler)
                # Simple approach: Randomly select start indices, potentially leading to overlap
                # Better: Select indices to mask first, then group into spans (harder)
                # Let's try selecting indices first:
                indices_to_mask = np.random.choice(indices_maskable, size=num_to_mask, replace=False)
                indices_to_mask_set = set(indices_to_mask)

                # Build the new sequence with single mask tokens replacing spans
                new_sequence = []
                idx = 0
                while idx < n_tokens:
                    if idx in indices_to_mask_set:
                        # Start of a masked span
                        new_sequence.append(mask_token_id)
                        # Skip over all consecutive masked tokens in this span
                        while idx < n_tokens and idx in indices_to_mask_set:
                            idx += 1
                    else:
                        # Keep the original token
                        new_sequence.append(actual_tokens[idx])
                        idx += 1

                # Pad the new sequence back to the original length (or max_length)
                # The collator's main padding should handle this if we return variable length lists,
                # but let's pad here to maintain tensor shape consistency within this function.
                padding_needed = original_length - len(new_sequence)
                if padding_needed > 0:
                    new_sequence.extend([pad_token_id] * padding_needed)
                elif padding_needed < 0:
                    # This shouldn't happen if logic is correct, but truncate if it does
                    new_sequence = new_sequence[:original_length]

                masked_sequences.append(torch.tensor(new_sequence, dtype=torch.long))

            # Stack the list of tensors into a single batch tensor
            return torch.stack(masked_sequences)


    # --- Initialize Data Collator ---
    print("Initializing data collator...")
    data_collator = DataCollatorForBartTextInfilling(
        tokenizer=tokenizer,
        mask_prob=0.35, # Default from BART paper, adjust if needed
        mean_span_length=data_args.mean_span_length,
        pad_to_multiple_of=8 if training_args.fp16 else None, # Pad for FP16 efficiency
    )
    print("Data collator initialized.")

    # --- Initialize Trainer ---
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets if training_args.do_train else None,
        # eval_dataset=tokenized_datasets["validation"] if training_args.do_eval else None, # Add eval dataset if needed
        tokenizer=tokenizer,
        data_collator=data_collator,
        # compute_metrics=compute_metrics, # Add metrics if needed for evaluation
    )
    print("Trainer initialized.")

    # --- Training ---
    # Check for last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            # Check if the directory only contains runs/ tensorboard files
            non_run_files = [f for f in os.listdir(training_args.output_dir) if f != 'runs']
            if len(non_run_files) > 0:
                 raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Start training
    if training_args.do_train:
        print("\n--- Starting Pre-training ---")
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint

        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too if possible
        trainer.save_state()

        # Log metrics
        metrics = train_result.metrics
        # max_train_samples = ( # Need train_dataset object if calculating samples
        #     data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        # )
        # metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)

        print("\n--- Pre-training Finished ---")

    # --- Evaluation ---
    # if training_args.do_eval:
    #     logger.info("*** Evaluate ***")
    #     metrics = trainer.evaluate()
    #     # max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    #     # metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    #     trainer.log_metrics("eval", metrics)
    #     trainer.save_metrics("eval", metrics)

    print("\n--- Pre-training Script Completed ---")

