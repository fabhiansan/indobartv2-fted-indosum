import argparse
import os
import logging
import time
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
    parser.add_argument(
        "--reuse_cache",
        action="store_true",
        help="Reuse existing prepared data if available instead of regenerating."
    )
    parser.add_argument(
        "--force_reload",
        action="store_true",
        help="Force reload dataset from source instead of using cached version."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    output_file_path = os.path.join(args.output_dir, args.output_filename)

    # Check if output file already exists and if reuse is enabled
    if os.path.exists(output_file_path) and args.reuse_cache:
        file_size = os.path.getsize(output_file_path)
        if file_size > 0:
            logger.info(f"Found existing output file at {output_file_path} ({file_size/1024/1024:.2f} MB)")
            logger.info(f"Skipping dataset preparation as --reuse_cache is enabled")
            return
        else:
            logger.warning(f"Found empty output file at {output_file_path}. Will recreate it.")
    
    logger.info(f"Loading dataset '{args.corpus_name}' with subset '{args.corpus_subset}'...")
    
    # Set up retry logic for network issues
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            # Load the dataset streamingly to avoid downloading everything at once if it's huge
            dataset = load_dataset(
                args.corpus_name, 
                args.corpus_subset, 
                split='train', 
                streaming=True, # Use streaming mode
                cache_dir=args.cache_dir,
                download_mode="force_redownload" if args.force_reload else None
            )
            logger.info("Dataset loaded successfully in streaming mode.")
            
            # Select a subset if max_samples is specified
            if args.max_samples:
                 logger.info(f"Processing a maximum of {args.max_samples} samples.")
                 dataset = dataset.take(args.max_samples)
            
            break  # Success, exit the retry loop
            
        except Exception as e:
            retry_count += 1
            if retry_count < max_retries:
                logger.warning(f"Attempt {retry_count} failed with error: {e}. Retrying...")
                time.sleep(5)  # Wait before retrying
            else:
                logger.error(f"Failed to load dataset after {max_retries} attempts: {e}")
                return

    logger.info(f"Writing processed text to {output_file_path}...")
    
    # Process and write the data line by line
    lines_written = 0
    empty_lines = 0
    invalid_lines = 0
    start_time = time.time()
    progress_interval = 10000  # Log progress every 10k lines
    
    try:
        with open(output_file_path, 'w', encoding='utf-8') as f:
            # Iterate through the dataset (streaming)
            # Wrap with tqdm for progress, note total might be unknown in streaming
            for i, example in enumerate(tqdm(dataset, desc="Processing dataset")): 
                # Check if 'text' field exists, try alternative fields if not
                text = None
                if 'text' in example:
                    text = example['text']
                elif 'content' in example:
                    text = example['content']
                elif len(example) > 0:
                    # Try the first field if it contains text
                    first_field = list(example.keys())[0]
                    if isinstance(example[first_field], str):
                        text = example[first_field]
                        if i == 0:  # Log this only for the first example
                            logger.warning(f"'text' field not found, using '{first_field}' instead.")
                
                if text and isinstance(text, str):
                    # Basic cleaning: strip whitespace, skip empty lines
                    cleaned_text = text.strip()
                    if cleaned_text:
                        f.write(cleaned_text + '\n')
                        lines_written += 1
                    else:
                        empty_lines += 1
                else:
                    invalid_lines += 1
                
                # Show progress periodically
                if lines_written > 0 and lines_written % progress_interval == 0:
                    elapsed = time.time() - start_time
                    rate = lines_written / elapsed if elapsed > 0 else 0
                    logger.info(f"Progress: {lines_written} lines written ({rate:.1f} lines/sec)")
                        
        # Final statistics
        elapsed_total = time.time() - start_time
        logger.info(f"Finished processing dataset:")
        logger.info(f"- Written: {lines_written} valid lines")
        logger.info(f"- Skipped: {empty_lines} empty lines, {invalid_lines} invalid items")
        logger.info(f"- Total time: {elapsed_total:.1f} seconds ({lines_written / elapsed_total:.1f} lines/sec)")
        logger.info(f"Output saved to: {output_file_path}")

        # Write a cache metadata file to help with future runs
        cache_metadata = {
            "timestamp": time.time(),
            "lines_count": lines_written,
            "corpus_name": args.corpus_name,
            "corpus_subset": args.corpus_subset,
            "max_samples": args.max_samples
        }
        
        # Serialize the metadata to a sidecar file
        try:
            import json
            metadata_path = output_file_path + ".meta.json"
            with open(metadata_path, 'w') as meta_file:
                json.dump(cache_metadata, meta_file, indent=2)
            logger.info(f"Cache metadata written to {metadata_path}")
        except Exception as e:
            logger.warning(f"Failed to write cache metadata: {e}")

    except Exception as e:
        logger.error(f"An error occurred during processing or writing: {e}")
        logger.error(f"Partial output may have been written to {output_file_path}")

if __name__ == "__main__":
    # Import time for timing operations
    import time
    main()
