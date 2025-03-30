import argparse
import os
import logging
from tokenizers import ByteLevelBPETokenizer
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Train a Byte-Level BPE tokenizer on a text corpus.")
    parser.add_argument(
        "--input_files", 
        nargs='+', 
        required=True, 
        help="List of path(s) to the input text file(s) (output from prepare_data.py)."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        required=True, 
        help="Directory to save the trained tokenizer files (vocab.json, merges.txt)."
    )
    parser.add_argument(
        "--vocab_size", 
        type=int, 
        default=50265, # Default often close to original BART
        help="Size of the vocabulary to train."
    )
    parser.add_argument(
        "--min_frequency", 
        type=int, 
        default=2, 
        help="Minimum frequency for tokens to be included in the vocabulary."
    )
    return parser.parse_args()

def main():
    args = parse_args()

    # Validate input files exist
    missing_files = []
    for file_path in args.input_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
            
    if missing_files:
        files_str = ", ".join(missing_files)
        logger.error(f"Input file(s) not found: {files_str}")
        logger.error("Please make sure all input files exist before training the tokenizer.")
        return

    # Check if any input files are empty
    empty_files = []
    for file_path in args.input_files:
        if os.path.getsize(file_path) == 0:
            empty_files.append(file_path)
            
    if empty_files:
        files_str = ", ".join(empty_files)
        logger.warning(f"The following input file(s) are empty: {files_str}")
        
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize a tokenizer
    # BART uses ByteLevelBPE
    logger.info("Initializing ByteLevelBPE tokenizer")
    tokenizer = ByteLevelBPETokenizer()

    # Log input files information
    total_size_mb = sum(os.path.getsize(f) for f in args.input_files) / (1024 * 1024)
    logger.info(f"Starting tokenizer training on {len(args.input_files)} file(s) ({total_size_mb:.2f} MB total)")
    logger.info(f"Vocabulary size: {args.vocab_size}, minimum token frequency: {args.min_frequency}")
    
    # Customize training
    start_time = time.time()
    try:
        tokenizer.train(
            files=args.input_files, 
            vocab_size=args.vocab_size, 
            min_frequency=args.min_frequency, 
            special_tokens=[
                "<s>",
                "<pad>",
                "</s>",
                "<unk>", 
                "<mask>",
            ])
        training_time = time.time() - start_time
        logger.info(f"Tokenizer training completed in {training_time:.2f} seconds.")

        # Save files to disk
        tokenizer_save_path = os.path.join(args.output_dir)
        tokenizer.save_model(tokenizer_save_path)
        logger.info(f"Tokenizer saved to {tokenizer_save_path} (vocab.json and merges.txt)")
        
        # Verify the saved files
        if os.path.exists(os.path.join(tokenizer_save_path, "vocab.json")) and \
           os.path.exists(os.path.join(tokenizer_save_path, "merges.txt")):
            logger.info("Successfully verified tokenizer files.")
        else:
            logger.warning("Tokenizer files may not have been saved correctly. Please check the output directory.")

    except (IOError, OSError) as e: 
        logger.error(f"I/O error during tokenizer training or saving: {e}", exc_info=True)
    except Exception as e: 
        logger.error(f"Unexpected error during tokenizer training or saving: {e}", exc_info=True)

if __name__ == "__main__":
    main()
