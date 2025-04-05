import argparse
import os
import logging
from tokenizers import ByteLevelBPETokenizer
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Train a ByteLevelBPE tokenizer.")
    parser.add_argument(
        "--input_files",
        nargs='+',  # Allows one or more input files
        required=True,
        help="Path(s) to the input text file(s) (e.g., cleaned_oscar_id.txt)."
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
        default=50265, # Standard BART vocab size
        help="Target vocabulary size."
    )
    parser.add_argument(
        "--min_frequency",
        type=int,
        default=2,
        help="Minimum frequency for a token to be included in the vocabulary."
    )
    args = parser.parse_args()

    # Expand user paths and ensure input files exist
    input_files_expanded = [os.path.expanduser(f) for f in args.input_files]
    output_dir_expanded = os.path.expanduser(args.output_dir)

    for f_path in input_files_expanded:
        if not os.path.isfile(f_path):
            logging.error(f"Input file not found: {f_path}")
            return

    # Ensure output directory exists
    Path(output_dir_expanded).mkdir(parents=True, exist_ok=True)

    logging.info(f"Initializing ByteLevelBPETokenizer...")
    # Initialize a tokenizer
    tokenizer = ByteLevelBPETokenizer()

    # Define special tokens standard for BART/RoBERTa style models
    special_tokens = [
        "<s>",   # Beginning of sentence
        "<pad>", # Padding
        "</s>",  # End of sentence
        "<unk>", # Unknown token
        "<mask>", # Mask token
    ]
    logging.info(f"Special tokens: {special_tokens}")

    logging.info(f"Starting tokenizer training...")
    logging.info(f"  Input files: {input_files_expanded}")
    logging.info(f"  Vocab size: {args.vocab_size}")
    logging.info(f"  Min frequency: {args.min_frequency}")

    # Customize training
    tokenizer.train(
        files=input_files_expanded,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=special_tokens
    )

    logging.info(f"Training complete. Saving tokenizer to {output_dir_expanded}")

    # Save files to output directory
    tokenizer.save_model(output_dir_expanded)

    logging.info(f"Tokenizer saved successfully.")

if __name__ == "__main__":
    main()
