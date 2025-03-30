import logging
import math
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import warnings
import packaging.version

# Disable wandb logging
os.environ["WANDB_DISABLED"] = "true"

import datasets
from datasets import load_dataset

import transformers
from transformers import (
    CONFIG_MAPPING,
    MODEL_FOR_MASKED_LM_MAPPING,
    AutoConfig,
    AutoModelForMaskedLM, # Using MLM for simplicity, BART would use Seq2Seq
    AutoModelForSeq2SeqLM,
    BartTokenizerFast, # Use the specific BART tokenizer
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
    DataCollatorForLanguageModeling,
    # Note: For true BART pre-training, a custom DataCollator implementing text infilling is needed.
    # DataCollatorForSeq2Seq could be a starting point but needs modification for denoising.
)
from transformers.trainer_utils import get_last_checkpoint

# Helper function (similar to what might be in transformers.utils)
def require_version(constraint, message):
    # Simplified check
    if packaging.version.parse(datasets.__version__) < packaging.version.parse("1.1.0"):
        warnings.warn("Streaming mode may require datasets>=1.1.0")

logger = logging.getLogger(__name__)
MODEL_CONFIG_CLASSES = list(MODEL_FOR_MASKED_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch."""
    model_name_or_path: Optional[str] = field(
        default="facebook/bart-base",
        metadata={
            "help": "The model checkpoint for weights initialization. Leave None if you want to train a model from scratch."
        },
    )
    model_type: Optional[str] = field(
        default="bart",
        metadata={"help": "If training from scratch, pass a model type from the list: " + ", ".join(MODEL_TYPES)},
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name_or_path"}
    )
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name_or_path"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to use as HTTP bearer authorization for remote files. If not specified, will use the token "
                "generated when running `huggingface-cli login` (stored in `~/.huggingface`)."
            )
        },
    )
    use_auth_token: Optional[bool] = field(
        default=None,
        metadata={
            "help": "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead."
        },
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether or not to allow for custom models defined on the Hub in their own modeling files. "
                "This option should only be set to `True` for repositories you trust and in which you have read the code, "
                "as it will execute code present on the Hub on your local machine."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split"
        },
    )
    max_seq_length: Optional[int] = field(
        default=512,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated." # Note: grouping handles this slightly differently
            )
        },
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    mlm_probability: float = field(
        default=0.15,
        metadata={"help": "Ratio of tokens to mask for masked language modeling loss (note: BART uses text infilling)"},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    streaming: bool = field(default=False, metadata={"help": "Enable streaming mode"})
    use_cached_prep: bool = field(
        default=False, 
        metadata={"help": "Use cached preprocessed datasets if available"}
    )
    dataset_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory to store cached datasets and preprocessed data"}
    )
    force_reload_raw: bool = field(
        default=False, 
        metadata={"help": "Force reload raw datasets even if cached version exists"}
    )
    save_preprocessed: bool = field(
        default=True, 
        metadata={"help": "Save preprocessed datasets to disk to speed up future runs"}
    )

    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."


@dataclass
class BartObjectiveArguments:
    """Arguments pertaining to the BART pre-training objective."""

    bart_objective: bool = field(
        default=False, metadata={"help": "Whether to use the BART text infilling objective instead of standard MLM."}
    )
    poisson_lambda: float = field(
        default=3.0, metadata={"help": "Lambda for Poisson distribution to sample span lengths for BART objective."}
    )
    masking_fraction: float = field(
        default=0.30, metadata={"help": "Fraction of tokens to mask for BART objective (note: BART paper used 0.3 of *spans*)."}
    )


@dataclass
class DataCollatorForBartSeq2Seq:
    """
    Data collator for BART sequence-to-sequence pre-training (text infilling).
    Reference: BART Paper (https://arxiv.org/abs/1910.13461)
    Adapated from similar implementations found online.
    """
    tokenizer: BartTokenizerFast
    masking_fraction: float # Fraction of tokens to mask
    poisson_lambda: float # Lambda for span length sampling
    pad_to_multiple_of: Optional[int] = None
    ignore_pad_token_for_loss: bool = True

    def __call__(self, examples):
        import torch
        import numpy as np
        import random
        
        # If batch is a list of dicts with 'text', process them
        if isinstance(examples[0], dict) and "text" in examples[0]:
            batch = self.tokenizer( # Tokenize the inputs
                [ex["text"] for ex in examples], 
                return_tensors="pt", 
                padding="longest", 
                truncation=True, 
                max_length=self.tokenizer.model_max_length
            )
        else:
            # Assume this is already a batch from the tokenized dataset
            batch = {
                k: torch.stack([torch.tensor(ex[k]) for ex in examples])
                if examples[0][k] is not None and isinstance(examples[0][k], list)
                else torch.tensor([ex[k] for ex in examples])
                for k in examples[0].keys()
            }
        
        input_ids = batch["input_ids"]
        batch["labels"] = input_ids.clone() # Labels are the original uncorrupted sequence

        # Perform BART's text infilling corruption
        masked_input_ids = input_ids.clone()
        
        # Get mask of special tokens to avoid masking them
        special_tokens_mask = batch.get("special_tokens_mask", None)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(ids, already_has_special_tokens=True)
                for ids in input_ids.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        
        # Create new attention masks and decoder input ids tensors
        attention_mask = batch.get("attention_mask", 
                           (input_ids != self.tokenizer.pad_token_id).long())
        
        # Create decoder input ids (BART uses shifted input for decoder)
        decoder_input_ids = torch.zeros_like(input_ids)
        
        # For each sequence in the batch
        for i in range(input_ids.size(0)):
            # Exclude padding and special tokens from potential masking
            seq_length = attention_mask[i].sum().item()
            if seq_length <= 2:  # Skip very short sequences (just BOS/EOS)
                continue
                
            non_special_indices = torch.nonzero(
                (~special_tokens_mask[i]) & 
                (input_ids[i] != self.tokenizer.pad_token_id),
                as_tuple=False
            ).squeeze()
            
            if len(non_special_indices) == 0:
                continue

            # BART paper masks ~30% of tokens
            n_tokens_to_mask = int(np.ceil(len(non_special_indices) * self.masking_fraction))
            if n_tokens_to_mask == 0:
                continue

            # Track which tokens are masked/deleted
            deleted_indices = set()
            
            # Keep selecting spans until we've masked enough tokens
            while len(deleted_indices) < n_tokens_to_mask:
                # Sample span length from Poisson distribution (λ=3)
                span_length = np.random.poisson(self.poisson_lambda)
                span_length = max(1, span_length)  # Ensure at least length 1
                
                # If we have too few tokens left to mask, just use what's available
                if len(deleted_indices) + span_length > n_tokens_to_mask:
                    span_length = n_tokens_to_mask - len(deleted_indices)
                    
                # Find available positions (not already masked/deleted)
                available_indices = [idx.item() for idx in non_special_indices 
                                   if idx.item() not in deleted_indices]
                if not available_indices:
                    break
                
                # Select random start position for the span
                start_idx = random.choice(available_indices)
                
                # Determine end of span (ensuring we don't go beyond sequence or mask too many tokens)
                end_idx = min(start_idx + span_length, seq_length - 1)
                
                # Check if we're not trying to mask special tokens
                span_indices = list(range(start_idx, end_idx))
                span_indices = [idx for idx in span_indices 
                              if not special_tokens_mask[i, idx] and idx not in deleted_indices]
                
                if not span_indices:
                    continue
                
                # Replace the first token with <mask>
                first_idx = span_indices[0]
                masked_input_ids[i, first_idx] = self.tokenizer.mask_token_id
                
                # Mark all tokens in span as deleted (except the first one we masked)
                for idx in span_indices[1:]:
                    deleted_indices.add(idx)
            
            # Now we need to actually delete the tokens marked for deletion
            # This is more complex as we need to shift tokens left
            if deleted_indices:
                deleted_indices = sorted(list(deleted_indices))
                new_seq = []
                
                # Keep only non-deleted tokens
                for j in range(seq_length):
                    if j not in deleted_indices:
                        new_seq.append(masked_input_ids[i, j].item())
                
                # Pad to original length
                new_seq = new_seq + [self.tokenizer.pad_token_id] * len(deleted_indices)
                
                # Update the masked sequence
                masked_input_ids[i, :seq_length] = torch.tensor(new_seq[:seq_length])
            
            # Create proper decoder input for sequence-to-sequence task
            # Shift the target right and prepend BOS token
            decoder_input_ids[i, 0] = self.tokenizer.bos_token_id
            decoder_input_ids[i, 1:seq_length] = batch["labels"][i, :seq_length-1]
        
        # Update with corrupted input ids and recalculate attention mask
        batch["input_ids"] = masked_input_ids
        batch["attention_mask"] = (masked_input_ids != self.tokenizer.pad_token_id).long()
        batch["decoder_input_ids"] = decoder_input_ids
        
        # Ignore padding tokens in labels
        if self.ignore_pad_token_for_loss:
            batch["labels"][batch["labels"] == self.tokenizer.pad_token_id] = -100

        return batch


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments, BartObjectiveArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args, bart_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args, bart_args = parser.parse_args_into_dataclasses()

    if model_args.use_auth_token is not None:
        warnings.warn(
            "The `use_auth_token` argument is deprecated and will be removed in v4.34. Please use `token` instead.",
            FutureWarning,
        )
        if model_args.token is not None:
            raise ValueError("`token` and `use_auth_token` are both specified. Please set only the argument `token`.")
        model_args.token = model_args.use_auth_token

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {training_args.parallel_mode.value == 'distributed'}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to info of the Transformers logger (on main process only):
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
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

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    preprocessed_dataset_path = None
    if data_args.use_cached_prep and data_args.dataset_cache_dir:
        cache_key = f"{model_args.model_name_or_path.replace('/', '_')}_{data_args.max_seq_length}"
        if data_args.train_file:
            cache_key += f"_{os.path.basename(data_args.train_file)}"
        elif data_args.dataset_name:
            cache_key += f"_{data_args.dataset_name}"
            if data_args.dataset_config_name:
                cache_key += f"_{data_args.dataset_config_name}"
                
        preprocessed_dataset_path = os.path.join(data_args.dataset_cache_dir, f"preprocessed_{cache_key}")
        
        if os.path.exists(preprocessed_dataset_path) and not data_args.overwrite_cache:
            try:
                logger.info(f"Loading preprocessed datasets from {preprocessed_dataset_path}")
                raw_datasets = datasets.load_from_disk(preprocessed_dataset_path)
                logger.info("Successfully loaded preprocessed datasets!")
                
                # Verify the datasets have the expected splits
                if "train" not in raw_datasets or (training_args.do_eval and "validation" not in raw_datasets):
                    logger.warning("Cached dataset doesn't have required splits. Reprocessing dataset.")
                    preprocessed_dataset_path = None  # Force reprocess
                else:
                    # Log dataset stats
                    logger.info(f"Train dataset size: {len(raw_datasets['train'])}")
                    if "validation" in raw_datasets:
                        logger.info(f"Validation dataset size: {len(raw_datasets['validation'])}")
            except Exception as e:
                logger.warning(f"Failed to load preprocessed datasets: {e}")
                preprocessed_dataset_path = None  # Force reprocess
    
    # If we don't have valid cached preprocessed datasets, load raw data
    if not data_args.use_cached_prep or preprocessed_dataset_path is None or data_args.overwrite_cache:
        logger.info("Loading raw datasets...")
        start_time = time.time()
        
        if data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset_name,
                data_args.dataset_config_name,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                streaming=data_args.streaming,
                download_mode="force_redownload" if data_args.force_reload_raw else None,
            )
            if "validation" not in raw_datasets.keys() and not data_args.streaming:
                 # Create validation split if not present
                raw_datasets["validation"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[:{data_args.validation_split_percentage}%]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    download_mode="force_redownload" if data_args.force_reload_raw else None,
                )
                raw_datasets["train"] = load_dataset(
                    data_args.dataset_name,
                    data_args.dataset_config_name,
                    split=f"train[{data_args.validation_split_percentage}%:]",
                    cache_dir=model_args.cache_dir,
                    token=model_args.token,
                    streaming=data_args.streaming,
                    download_mode="force_redownload" if data_args.force_reload_raw else None,
                )
            elif "validation" not in raw_datasets.keys() and data_args.streaming:
                 logger.warning("Streaming mode detected without validation split. Validation will be skipped.")
                 # Cannot easily split in streaming mode without loading everything
                 raw_datasets["validation"] = None 

        else:
            data_files = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
            if hasattr(data_args, 'test_file') and data_args.test_file is not None:
                data_files["test"] = data_args.test_file

            # Determine the extension from the train file if available, otherwise validation
            first_file = data_args.train_file or data_args.validation_file
            if not first_file:
                 raise ValueError("Need at least a train_file or validation_file when dataset_name is not provided.")
            extension = first_file.split(".")[-1]
            if extension == "txt":
                extension = "text" # datasets library uses 'text' for .txt files
                if hasattr(data_args, 'keep_linebreaks'):
                    dataset_args = {"keep_linebreaks": data_args.keep_linebreaks}
                else:
                    dataset_args = {}
            else:
                dataset_args = {}

            raw_datasets = load_dataset(
                extension, # Pass the file type ('text', 'csv', 'json')
                data_files=data_files,
                cache_dir=model_args.cache_dir,
                token=model_args.token,
                **dataset_args,
            )
            # If no validation data is there, validation_split_percentage will be used to divide the dataset.
            if "validation" not in raw_datasets.keys():
                if data_args.validation_split_percentage is None or data_args.validation_split_percentage == 0:
                    raise ValueError(
                        f"Need either a validation file or a validation_split_percentage greater than 0."
                    )
                
                raw_datasets = raw_datasets["train"].train_test_split(
                    test_size=data_args.validation_split_percentage / 100
                )
                raw_datasets["train"] = raw_datasets["train"]
                raw_datasets["validation"] = raw_datasets["test"]
                del raw_datasets["test"]
                
        logger.info(f"Raw datasets loaded in {time.time() - start_time:.2f} seconds")

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.config_name:
        config = AutoConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    else:
        config = CONFIG_MAPPING[model_args.model_type]()
        logger.warning("You are instantiating a new config instance from scratch.")

    tokenizer_kwargs = {
        "cache_dir": model_args.cache_dir,
        "use_fast": model_args.use_fast_tokenizer,
        "revision": model_args.model_revision,
        "token": model_args.token,
        "trust_remote_code": model_args.trust_remote_code,
    }
    if model_args.tokenizer_name_or_path:
         # Load the custom trained tokenizer (expects vocab.json and merges.txt in the dir)
        tokenizer = BartTokenizerFast.from_pretrained(model_args.tokenizer_name_or_path, **tokenizer_kwargs)
        logger.info(f"Loaded custom tokenizer from {model_args.tokenizer_name_or_path}")
    elif model_args.model_name_or_path:
         # This path would load the default tokenizer for the model, 
         # which is usually not what we want when pre-training with a custom vocab
        # tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, **tokenizer_kwargs)
        raise ValueError(
             "You are instantiating a new tokenizer from scratch. This is not supported by this script." 
             "Provide the path to your trained tokenizer with --tokenizer_name_or_path."
         )
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it here via --tokenizer_name_or_path."
        )

    if model_args.model_name_or_path:
        # Load the base BART model
        # Note: We load BartForConditionalGeneration as BART is inherently seq2seq
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            token=model_args.token,
            trust_remote_code=model_args.trust_remote_code,
        )
    else:
        logger.info("Training new model from scratch")
        model = AutoModelForSeq2SeqLM.from_config(config, trust_remote_code=model_args.trust_remote_code)

    # Resize token embeddings if necessary
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != embedding_size:
        logger.info(f"Resizing model token embeddings from {embedding_size} to {len(tokenizer)}")
        model.resize_token_embeddings(len(tokenizer))

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    if training_args.do_train:
        # Check if raw_datasets is defined
        if 'raw_datasets' not in locals() or raw_datasets is None:
            raise ValueError(
                "No dataset is available for training. Please provide a dataset_name or train_file."
            )
        column_names = list(raw_datasets["train"].features) if not data_args.streaming else list(raw_datasets["train"].take(1))[0].keys()
    else:
         if data_args.validation_file is not None:
             # Check if raw_datasets is defined
             if 'raw_datasets' not in locals() or raw_datasets is None:
                 raise ValueError(
                     "No dataset is available for validation. Please provide a dataset_name or validation_file."
                 )
             column_names = list(raw_datasets["validation"].features) if not data_args.streaming else list(raw_datasets["validation"].take(1))[0].keys()
         else:
             # Handle case where only evaluation is done and train_file was used for split
             # Check if raw_datasets is defined
             if 'raw_datasets' not in locals() or raw_datasets is None:
                 raise ValueError(
                     "No dataset is available. Please provide a dataset_name, train_file, or validation_file."
                 )
             column_names = list(raw_datasets["train"].features) if not data_args.streaming else list(raw_datasets["train"].take(1))[0].keys()

    text_column_name = "text" if "text" in column_names else column_names[0]

    if data_args.line_by_line:
        # When using line_by_line, we just tokenize each nonempty line.
        padding = "max_length" if data_args.pad_to_max_length else False

        def tokenize_function(examples):
            # Remove empty lines
            examples[text_column_name] = [
                line for line in examples[text_column_name] if len(line) > 0 and not line.isspace()
            ]
            return tokenizer(
                examples[text_column_name],
                padding=padding,
                truncation=True,
                max_length=data_args.max_seq_length,
                # We use this option because DataCollatorForLanguageModeling relies on it.
                return_special_tokens_mask=True,
            )
        
        with training_args.main_process_first(desc="dataset map tokenization"):
            tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=[text_column_name],
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on dataset line_by_line",
            )
    else:
        # Otherwise, we tokenize every text, then concatenate them together before splitting
        # them in smaller parts. We use `return_special_tokens_mask=True` because DataCollatorForLanguageModeling
        # is more efficient when it receives the `special_tokens_mask`.
        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        with training_args.main_process_first(desc="dataset map tokenization"):
             if not data_args.streaming:
                tokenized_datasets = raw_datasets.map(
                    tokenize_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on dataset",
                )
             else:
                 tokenized_datasets = raw_datasets.map(
                     tokenize_function,
                     batched=True,
                     remove_columns=column_names,
                 )

        # Main data processing function that will concatenate all texts from our dataset and generate chunks of
        # max_seq_length.
        block_size = data_args.max_seq_length

        def group_texts(examples):
            # Concatenate all texts.
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            # We drop the small remainder, we could add padding if the model supported it instead of this drop,
            # you can customize this part to your needs.
            if total_length >= block_size:
                total_length = (total_length // block_size) * block_size
            # Split by chunks of max_len.
            result = {
                k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
                for k, t in concatenated_examples.items()
            }
            return result
        
        # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a
        # remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value
        # might be slower to preprocess. To speed up this part, we use multiprocessing.
        with training_args.main_process_first(desc="grouping texts together"):
             if not data_args.streaming:
                tokenized_datasets = tokenized_datasets.map(
                    group_texts,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc=f"Grouping texts in chunks of {block_size}",
                )
             else:
                 tokenized_datasets = tokenized_datasets.map(
                     group_texts, batched=True
                 )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = tokenized_datasets["train"]
        if data_args.max_train_samples is not None:
             max_train_samples = min(len(train_dataset), data_args.max_train_samples)
             train_dataset = train_dataset.select(range(max_train_samples)) if not data_args.streaming else train_dataset.take(max_train_samples) 
             

    if training_args.do_eval:
         # Use validation split created earlier or the provided validation file
        if "validation" not in tokenized_datasets or tokenized_datasets["validation"] is None:
             logger.warning("Validation dataset not found or could not be created. Skipping evaluation.")
             training_args.do_eval = False # Disable eval if no validation set
             eval_dataset = None
        else:
             eval_dataset = tokenized_datasets["validation"]
             if data_args.max_eval_samples is not None:
                 max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
                 eval_dataset = eval_dataset.select(range(max_eval_samples)) if not data_args.streaming else eval_dataset.take(max_eval_samples)

    # Data collator
    # We have already loaded the tokenizer, so we use it directly.
    if bart_args.bart_objective:
        logger.info("Using BART text infilling objective with custom data collator.")
        # Ensure tokenizer has mask token if we are doing BART objective
        if tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is required for the BART objective. "
                "Make sure you are using a model suitable for this task or add a mask token during tokenizer training/loading."
            )
        data_collator = DataCollatorForBartSeq2Seq(
            tokenizer=tokenizer,
            masking_fraction=bart_args.masking_fraction,
            poisson_lambda=bart_args.poisson_lambda,
            pad_to_multiple_of=8 if training_args.fp16 else None, # Pad for FP16 efficiency
        )
    else:
        logger.info("Using standard Masked Language Modeling (MLM) objective.")
        if tokenizer.mask_token is None:
             warnings.warn(
                "This tokenizer does not have a mask token which is required for MLM. Adding one temporarily."
                " Consider adding it permanently during tokenizer training/loading."
             )
             # Potentially add a mask token here if really needed, but it's better if the tokenizer has it.
             # tokenizer.add_special_tokens({'mask_token': '[MASK]'}) 
             # model.resize_token_embeddings(len(tokenizer)) 

        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm_probability=data_args.mlm_probability,
            pad_to_multiple_of=8 if training_args.fp16 else None, # Pad for FP16 efficiency
        )

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics

        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate()

        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card or push to hub logic can be added here
    # kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "fill-mask"}
    # if data_args.dataset_name is not None:
    #     kwargs["dataset_tags"] = data_args.dataset_name
    #     if data_args.dataset_config_name is not None:
    #         kwargs["dataset_args"] = data_args.dataset_config_name
    #         kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
    #     else:
    #         kwargs["dataset"] = data_args.dataset_name

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)

    # Log a few random samples from the training set:
    if training_args.do_train:
        logger.info(f"Tokenization complete in {time.time() - tokenization_start_time:.2f} seconds")
        
        # Save preprocessed datasets to disk if requested
        if data_args.save_preprocessed and data_args.dataset_cache_dir and not data_args.streaming:
            try:
                os.makedirs(data_args.dataset_cache_dir, exist_ok=True)
                logger.info(f"Saving preprocessed datasets to {preprocessed_dataset_path}")
                tokenized_datasets.save_to_disk(preprocessed_dataset_path)
                
                # Save metadata about this preprocessing
                with open(os.path.join(preprocessed_dataset_path, "preprocessing_info.json"), "w") as f:
                    json.dump({
                        "timestamp": time.time(),
                        "model_name": model_args.model_name_or_path,
                        "max_seq_length": data_args.max_seq_length,
                        "padding": "max_length" if data_args.pad_to_max_length else "longest",
                        "line_by_line": data_args.line_by_line,
                        "mlm_probability": data_args.mlm_probability,
                    }, f, indent=2)
                logger.info("Preprocessing data cached successfully")
            except Exception as e:
                logger.warning(f"Failed to save preprocessed datasets: {e}")
        
        for index in random.sample(range(len(tokenized_datasets["train"])), min(3, len(tokenized_datasets["train"]))):
            logger.info(f"Sample {index} of the training set:")
            for key in tokenized_datasets["train"][index]:
                if key != "attention_mask":  # Skip attention mask for brevity
                    value = tokenized_datasets["train"][index][key]
                    if isinstance(value, list) and len(value) > 20:
                        value = str(value[:20]) + "... (truncated)"
                    logger.info(f"  {key}: {value}")

if __name__ == "__main__":
    main()
