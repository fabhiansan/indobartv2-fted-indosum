import argparse
import os
from datasets import load_dataset
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Cleaning Functions ---
MIN_DOC_LENGTH = 100 # Minimum characters for a document to be kept

def is_valid_doc(example):
    """Checks if a document is long enough."""
    return len(example['text']) > MIN_DOC_LENGTH

# --- Main Script ---
def main():
    parser = argparse.ArgumentParser(description="Load, clean, and format OSCAR dataset from cache.")
    parser.add_argument(
        "--cache_path",
        type=str,
        required=True,
        help="Path to the specific dataset cache directory (e.g., ~/.cache/huggingface/datasets/oscar-corpus___oscar-2301/id)."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the cleaned text output file (e.g., cleaned_oscar_id.txt)."
    )
    args = parser.parse_args()

    # Expand user path if necessary
    cache_path_expanded = os.path.expanduser(args.cache_path)
    output_file_expanded = os.path.expanduser(args.output_file)

    if not os.path.isdir(cache_path_expanded):
        logging.error(f"Cache directory not found: {cache_path_expanded}")
        return

    logging.info(f"Loading dataset from cache: {cache_path_expanded}")
    # Note: Loading directly from a specific cache path like this might require
    # the cache to be structured exactly as load_dataset expects after download.
    # If this fails, alternative methods like load_from_disk might be needed
    # depending on the exact cache structure.
    # Assuming the path points to the directory containing Arrow files or similar.
    try:
        # We might need to load specific splits if they exist, e.g., 'train'
        # Trying to load whatever is directly in the path first.
        dataset = load_dataset(cache_path_expanded, split='train') # Adjust split name if needed
        logging.info(f"Dataset loaded successfully. Number of documents: {len(dataset)}")
    except Exception as e:
        logging.error(f"Failed to load dataset from {cache_path_expanded}. Error: {e}")
        logging.error("Please ensure the path points to a valid Hugging Face dataset cache directory containing the data files (e.g., Arrow format).")
        return

    logging.info(f"Applying cleaning filters (min length: {MIN_DOC_LENGTH})...")
    # Apply cleaning - filter short documents
    cleaned_dataset = dataset.filter(is_valid_doc, num_proc=os.cpu_count()) # Use multiple processes
    logging.info(f"Dataset size after cleaning: {len(cleaned_dataset)}")

    logging.info(f"Writing cleaned text to {output_file_expanded}...")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file_expanded), exist_ok=True)

    count = 0
    with open(output_file_expanded, 'w', encoding='utf-8') as f:
        for example in cleaned_dataset:
            # Write each document's text, followed by a newline
            f.write(example['text'].strip() + '\n')
            count += 1
            if count % 10000 == 0:
                logging.info(f"Written {count} documents...")

    logging.info(f"Finished writing {count} cleaned documents to {output_file_expanded}")

if __name__ == "__main__":
    main()
