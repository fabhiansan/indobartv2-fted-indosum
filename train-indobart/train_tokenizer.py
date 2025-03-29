import argparse
import os
import logging
from tokenizers import ByteLevelBPETokenizer

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

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize a tokenizer
    # BART uses ByteLevelBPE
    tokenizer = ByteLevelBPETokenizer()

    # Customize training
    logger.info("Starting tokenizer training with vocab_size=%d, min_frequency=%d...", args.vocab_size, args.min_frequency)
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
        logger.info("Tokenizer training completed.")

        # Save files to disk
        tokenizer_save_path = os.path.join(args.output_dir)
        tokenizer.save_model(tokenizer_save_path)
        logger.info("Tokenizer files saved to %s", tokenizer_save_path)

    except (IOError, OSError) as e:
        logger.error("An I/O error occurred during tokenizer training or saving: %s", e, exc_info=True)
    except Exception as e: # Catch any other unexpected exception
        logger.error("An error occurred during tokenizer training or saving: %s", e, exc_info=True)

if __name__ == "__main__":
    main()
