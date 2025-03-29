import argparse
import os
import logging
from datasets import load_dataset
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare Indonesian text corpus for pre-training.")
    parser.add_argument(
        "--corpus_name", 
        type=str, 
        default="oscar", 
        help="Name of the corpus to download from Hugging Face datasets (e.g., 'oscar')."
    )
    parser.add_argument(
        "--corpus_subset", 
        type=str, 
        default="unshuffled_deduplicated_id", 
        help="Subset or configuration of the corpus (e.g., 'unshuffled_deduplicated_id' for OSCAR Indonesian)."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./data", 
        help="Directory to save the prepared text file."
    )
    parser.add_argument(
        "--output_filename", 
        type=str, 
        default="indonesian.txt", 
        help="Name for the output text file."
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="Directory to cache downloaded datasets."
    )
    parser.add_argument(
        "--num_proc",
        type=int,
        default=None, # Use default number of processors
        help="Number of processes to use for dataset processing."
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None, # Process all samples by default
        help="Maximum number of text samples to process (for testing/debugging)."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, args.output_filename)

    logger.info(f"Loading dataset '{args.corpus_name}' with subset '{args.corpus_subset}'...")
    try:
        # Load the dataset streamingly to avoid downloading everything at once if it's huge
        dataset = load_dataset(
            args.corpus_name, 
            args.corpus_subset, 
            split='train', 
            streaming=True, # Use streaming mode
            cache_dir=args.cache_dir
        )
        logger.info("Dataset loaded successfully in streaming mode.")
        
        # Select a subset if max_samples is specified
        if args.max_samples:
             logger.info(f"Processing a maximum of {args.max_samples} samples.")
             dataset = dataset.take(args.max_samples)

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        return

    logger.info(f"Writing processed text to {output_file_path}...")
    
    # Process and write the data line by line
    lines_written = 0
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # Iterate through the dataset (streaming)
            # Wrap with tqdm for progress, note total might be unknown in streaming
            for example in tqdm(dataset, desc="Processing dataset"): 
                text = example.get('text', '') # Assuming the text field is named 'text'
                if text and isinstance(text, str):
                    # Basic cleaning: strip whitespace, skip empty lines
                    cleaned_text = text.strip()
                    if cleaned_text:
                        f.write(cleaned_text + '\n')
                        lines_written += 1
                        
        logger.info(f"Finished writing {lines_written} lines to {output_file_path}.")

    except Exception as e:
        logger.error(f"An error occurred during processing or writing: {e}")

if __name__ == "__main__":
    main()
