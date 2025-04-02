import argparse
import os
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer

# --- Configuration ---
# Adjust these parameters as needed
DATASET_NAME = "oscar-corpus/OSCAR-2301" # Or OSCAR-2201, etc.
DATASET_LANG = "id"
VOCAB_SIZE = 32000 # Example vocabulary size
MIN_FREQUENCY = 2
OUTPUT_DIR = "./indonesian_tokenizer"
# Set this if you need to authenticate with Hugging Face Hub
# Requires `huggingface-cli login` beforehand or setting HF_TOKEN env var
USE_AUTH_TOKEN = True # Set to False if the dataset doesn't require login

# --- Argument Parsing ---
# Allows overriding config via command line
parser = argparse.ArgumentParser(description="Train a BPE tokenizer on Indonesian OSCAR data.")
parser.add_argument("--dataset_name", type=str, default=DATASET_NAME, help="Hugging Face dataset name (e.g., oscar-corpus/OSCAR-2301)")
parser.add_argument("--dataset_lang", type=str, default=DATASET_LANG, help="Language code for the dataset (e.g., 'id')")
parser.add_argument("--vocab_size", type=int, default=VOCAB_SIZE, help="Target vocabulary size")
parser.add_argument("--min_frequency", type=int, default=MIN_FREQUENCY, help="Minimum frequency for tokens")
parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR, help="Directory to save the tokenizer files")
parser.add_argument("--use_auth_token", type=bool, default=USE_AUTH_TOKEN, help="Use authentication token for private/gated datasets")
args = parser.parse_args()

# --- Main Script ---
def get_training_corpus(dataset_stream):
    """
    Generator function to yield text data from the dataset stream.
    Handles potential errors and yields batches of text.
    """
    batch_size = 1000
    count = 0
    while True:
        batch = []
        try:
            for _ in range(batch_size):
                item = next(dataset_stream)
                # Assuming the text content is in a field named 'text' or 'content'
                # Adjust field name if necessary based on the dataset structure
                text_content = item.get('text', item.get('content'))
                if text_content:
                    batch.append(text_content)
                count += 1
            if not batch:
                break # No more data
            yield batch
            print(f"Processed {count} documents...", end='\r')
        except StopIteration:
            print("\nReached end of dataset stream.")
            if batch: # Yield any remaining items
                 yield batch
            break
        except Exception as e:
            print(f"\nError reading dataset stream: {e}")
            # Optionally continue or break based on error handling strategy
            break

if __name__ == "__main__":
    print("--- Starting Tokenizer Training ---")
    print(f"Dataset: {args.dataset_name}, Language: {args.dataset_lang}")
    print(f"Vocab Size: {args.vocab_size}, Min Frequency: {args.min_frequency}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Use Auth Token: {args.use_auth_token}")

    # 1. Initialize Tokenizer
    # Using ByteLevelBPETokenizer, suitable for many models including BART
    tokenizer = ByteLevelBPETokenizer()

    # 2. Load Dataset (Streaming)
    print("\nLoading dataset stream...")
    # Use streaming=True to avoid downloading the entire dataset at once
    # Requires authentication if the dataset is gated
    try:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_lang,
            split="train",
            streaming=True,
            use_auth_token=args.use_auth_token
        )
        dataset_iterator = iter(dataset)
    except Exception as e:
        print(f"\nError loading dataset: {e}")
        print("Please ensure you are logged in via `huggingface-cli login` if needed.")
        exit(1)

    # 3. Train Tokenizer
    print("\nTraining tokenizer...")
    # The train_from_iterator function expects an iterator that yields lists of strings
    tokenizer.train_from_iterator(
        get_training_corpus(dataset_iterator),
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=[
            "<s>", # Beginning of Sequence (for BART)
            "<pad>", # Padding
            "</s>", # End of Sequence / Separator (for BART)
            "<unk>", # Unknown
            "<mask>", # Mask token (for BART pre-training)
        ],
        show_progress=True,
    )
    print("\nTokenizer training complete.")

    # 4. Save Tokenizer
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        print(f"Created output directory: {args.output_dir}")

    tokenizer.save_model(args.output_dir)
    print(f"Tokenizer files saved to {args.output_dir}")
    print("Files created: vocab.json, merges.txt")

    print("\n--- Tokenizer Training Finished ---")
    logger.info("Script finished successfully.")

# Example Usage (from command line):
# python train_tokenizer.py --vocab_size 32000 --output_dir ./my_indo_tokenizer
