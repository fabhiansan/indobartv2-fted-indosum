# data_utils.py

import logging
from datasets import load_dataset, concatenate_datasets, DatasetDict, Features, Value, Dataset
from transformers import AutoTokenizer
import finetune_config as cfg
import pyarrow.feather as feather
import pyarrow as pa
import pandas as pd
import glob
import os
import os

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
        truncation=True
        # Padding will be handled dynamically by the DataCollatorForSeq2Seq
    )

    # Setup the tokenizer for targets
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            examples["target_text"],
            max_length=cfg.MAX_TARGET_LENGTH,
            truncation=True
            # Padding will be handled dynamically by the DataCollatorForSeq2Seq
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
            # Check if a local path is specified for this dataset
            local_path = cfg.LOCAL_DATA_PATHS.get(name)
            config_name = 'canonical' if name == 'liputan6' else None # Keep config name logic

            if local_path and name == 'liputan6': # Specific logic for local Liputan6
                logging.info(f"Attempting to load {name} (Arrow format) from local path: {local_path}")
                try:
                    # Define data files assuming train/validation/test subdirs
                    data_files = {
                        "train": os.path.join(local_path, "train/*.arrow"),
                        "validation": os.path.join(local_path, "validation/*.arrow"),
                        "test": os.path.join(local_path, "test/*.arrow")
                    }
                    # Basic check - adjust if needed based on actual file structure/names
                    import glob
                    if not glob.glob(data_files["train"]):
                         raise FileNotFoundError(f"No train arrow files found in {os.path.join(local_path, 'train')}")

                    raw_dataset = load_dataset("arrow", data_files=data_files)
                    logging.info(f"Successfully loaded {name} from local Arrow files.")
                except Exception as arrow_load_err:
                    logging.error(f"Failed to load {name} from local Arrow files at {local_path}.", exc_info=True)
                    raise arrow_load_err
            elif local_path and name == 'indosum': # Specific logic for local IndoSUM using pyarrow
                logging.info(f"Attempting to load {name} (Arrow format) manually from local path: {local_path}")
                try:
                    split_datasets = {}
                    # Define expected features based on config (corrected column names)
                    expected_features = Features({
                        cfg.DATASET_COLUMNS['indosum']['document']: Value('string'),
                        cfg.DATASET_COLUMNS['indosum']['summary']: Value('string')
                    })

                    for split_name_config, subdir_name in [("train", "traindataset"), ("validation", "devdataset"), ("test", "testdataset")]:
                        split_path_pattern = os.path.join(local_path, subdir_name, "*.arrow")
                        arrow_files = glob.glob(split_path_pattern)
                        if not arrow_files:
                            logging.warning(f"No arrow files found for split '{split_name_config}' in {os.path.join(local_path, subdir_name)}. Skipping this split for IndoSUM.")
                            continue

                        logging.info(f"Loading {len(arrow_files)} arrow file(s) for IndoSUM split '{split_name_config}' from {os.path.join(local_path, subdir_name)}")
                        # Load and concatenate tables if multiple files exist for a split
                        tables = [feather.read_table(f) for f in arrow_files]
                        pa_table = pa.concat_tables(tables) if tables else None

                        if pa_table:
                            # Create dataset directly from pyarrow table with defined features
                            split_datasets[split_name_config] = Dataset.from_arrow(pa_table, features=expected_features)
                        else:
                             logging.warning(f"No data loaded for IndoSUM split '{split_name_config}'.")


                    if not split_datasets:
                         raise ValueError("No splits could be loaded for local IndoSUM dataset.")

                    raw_dataset = DatasetDict(split_datasets)
                    logging.info(f"Successfully loaded {name} from local Arrow files using pyarrow.")

                except Exception as manual_arrow_load_err:
                    logging.error(f"Failed to load {name} manually from local Arrow files at {local_path}.", exc_info=True)
                    raise manual_arrow_load_err
            elif local_path: # Handle other potential local datasets generically
                 logging.warning(f"Generic local path loading not fully implemented for {name}. Trying standard load from path.")
                 raw_dataset = load_dataset(local_path, trust_remote_code=True)
            else:
                # Load from Hugging Face Hub if no local path specified or not liputan6
                logging.info(f"Attempting to load {name} ({path}) from Hub with trust_remote_code=True and config: {config_name}")
                raw_dataset = load_dataset(path, name=config_name, trust_remote_code=True)
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