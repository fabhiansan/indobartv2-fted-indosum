# data_utils.py

import logging
from datasets import load_dataset, concatenate_datasets, DatasetDict
from transformers import AutoTokenizer
import finetune_config as cfg

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verify_and_rename_columns(dataset, expected_cols, dataset_name):
    """Renames columns to 'input_text' and 'target_text'."""
    current_cols = dataset.column_names
    doc_col = expected_cols.get("document")
    sum_col = expected_cols.get("summary")

    if doc_col not in current_cols or sum_col not in current_cols:
        raise ValueError(
            f"Dataset '{dataset_name}' does not have expected columns '{doc_col}' and '{sum_col}'. "
            f"Found columns: {current_cols}. Please check DATASET_COLUMNS in finetune_config.py."
        )

    # Only rename if necessary
    rename_map = {}
    if doc_col != "input_text":
        rename_map[doc_col] = "input_text"
    if sum_col != "target_text":
        rename_map[sum_col] = "target_text"

    if rename_map:
        logging.info(f"Renaming columns for {dataset_name}: {rename_map}")
        dataset = dataset.rename_columns(rename_map)

    # Keep only the necessary columns
    final_cols = ["input_text", "target_text"]
    cols_to_remove = [col for col in dataset.column_names if col not in final_cols]
    if cols_to_remove:
        dataset = dataset.remove_columns(cols_to_remove)

    return dataset


def preprocess_function(examples, tokenizer):
    """Tokenizes inputs and targets."""
    inputs = [cfg.SUMMARIZATION_PREFIX + doc for doc in examples["input_text"]]
    model_inputs = tokenizer(
        inputs,
        max_length=cfg.MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length" # Pad later with collator if preferred
    )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=cfg.MAX_TARGET_LENGTH,
            truncation=True,
            padding="max_length" # Pad later with collator if preferred
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def load_and_prepare_datasets(tokenizer):
    """Loads, preprocesses, and combines datasets specified in the config."""
    processed_datasets = {"train": [], "validation": [], "test": []}
    split_mapping = {"train": "train", "validation": "validation", "test": "test"}

    # Special handling for xsum validation/test splits
    xsum_split_mapping = {"train": "train", "validation": "validation", "test": "test"}
    # Liputan6 might only have train/test, use test for validation if needed
    liputan6_split_mapping = {"train": "train", "validation": "test", "test": "test"}
    # IndoSUM might only have train/test, use test for validation if needed
    indosum_split_mapping = {"train": "train", "validation": "test", "test": "test"}


    dataset_split_mappings = {
        "xsum": xsum_split_mapping,
        "indosum": indosum_split_mapping,
        "liputan6": liputan6_split_mapping
    }


    for name, path in cfg.DATASET_NAMES.items():
        logging.info(f"Loading dataset: {name} ({path})")
        try:
            # Always trust remote code for these datasets as they might have custom scripts
            logging.info(f"Attempting to load {path} with trust_remote_code=True")
            raw_dataset = load_dataset(path, trust_remote_code=True)
            logging.info(f"Raw dataset '{name}' loaded. Features: {raw_dataset}")

            current_split_map = dataset_split_mappings.get(name, split_mapping)

            for split_type in ["train", "validation", "test"]:
                hf_split_name = current_split_map.get(split_type)
                if hf_split_name not in raw_dataset:
                    logging.warning(f"Split '{hf_split_name}' not found in dataset '{name}'. Skipping {split_type} split.")
                    continue

                logging.info(f"Processing {split_type} split for {name} (using HF split: {hf_split_name})")
                split_data = raw_dataset[hf_split_name]

                # Verify and rename columns
                expected_cols = cfg.DATASET_COLUMNS.get(name)
                if not expected_cols:
                     raise ValueError(f"Column mapping not defined for dataset '{name}' in finetune_config.py")
                renamed_data = verify_and_rename_columns(split_data, expected_cols, name)

                # Apply subsetting if configured
                max_samples = None
                if split_type == "train" and cfg.MAX_TRAIN_SAMPLES is not None:
                    max_samples = min(cfg.MAX_TRAIN_SAMPLES, len(renamed_data))
                elif split_type == "validation" and cfg.MAX_EVAL_SAMPLES is not None:
                     max_samples = min(cfg.MAX_EVAL_SAMPLES, len(renamed_data))
                elif split_type == "test" and cfg.MAX_EVAL_SAMPLES is not None: # Use eval limit for test too
                     max_samples = min(cfg.MAX_EVAL_SAMPLES, len(renamed_data))

                if max_samples is not None:
                    logging.info(f"Subsetting {name} {split_type} split to {max_samples} samples.")
                    renamed_data = renamed_data.select(range(max_samples))

                # Tokenize
                logging.info(f"Tokenizing {name} {split_type} split...")
                tokenized_split = renamed_data.map(
                    lambda examples: preprocess_function(examples, tokenizer),
                    batched=True,
                    remove_columns=["input_text", "target_text"] # Remove original text columns
                )
                processed_datasets[split_type].append(tokenized_split)
                logging.info(f"Finished processing {split_type} split for {name}.")

        except Exception as e:
            # Special handling for potential IndoSUM generation error
            if name == "indosum" and isinstance(e, ValueError) and "NotADirectoryError" in str(e):
                 logging.warning(f"Skipping dataset '{name}' due to a potential generation/caching issue: {e}", exc_info=False)
            # Handle general loading/processing errors
            else:
                logging.error(f"Failed to load or process dataset {name}: {e}", exc_info=True)
            # Continue to the next dataset instead of stopping execution
            continue # Skip the rest of the loop for this failed dataset

    # Combine datasets for each split
    final_datasets = DatasetDict()
    for split_type, ds_list in processed_datasets.items():
        if ds_list:
            logging.info(f"Concatenating {len(ds_list)} datasets for the {split_type} split.")
            final_datasets[split_type] = concatenate_datasets(ds_list).shuffle(seed=42) # Shuffle combined dataset
            logging.info(f"Final {split_type} dataset size: {len(final_datasets[split_type])}")
        else:
             logging.warning(f"No datasets were successfully processed for the {split_type} split.")


    if not final_datasets:
        raise RuntimeError("No datasets could be loaded or processed. Check configuration and dataset availability.")

    return final_datasets

if __name__ == '__main__':
    # Example usage: Load tokenizer and prepare datasets
    logging.info("Running data_utils.py standalone example...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(cfg.BASE_MODEL_NAME)
        logging.info(f"Tokenizer loaded: {cfg.BASE_MODEL_NAME}")
        prepared_datasets = load_and_prepare_datasets(tokenizer)
        logging.info("Datasets prepared successfully.")
        print("\nPrepared Dataset Splits:")
        for split_name, dataset in prepared_datasets.items():
            print(f"- {split_name}: {len(dataset)} samples")
            print(f"  Features: {dataset.features}\n")
    except Exception as e:
        logging.error(f"Error in standalone example: {e}", exc_info=True)