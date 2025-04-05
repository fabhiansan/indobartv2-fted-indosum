import argparse
import logging
import math
import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (
    BartForConditionalGeneration,
    BartTokenizerFast,
    DataCollatorForLanguageModeling, # Can use MLM collator for perplexity calculation
)
from tqdm import tqdm

# Setup logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity of a pre-trained BART model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the directory containing the pre-trained model and tokenizer.",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        required=True,
        help="Path to the test text file (plain text, one sequence per line).",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for evaluation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (cuda or cpu).",
    )
    args = parser.parse_args()

    model_path_expanded = os.path.expanduser(args.model_path)
    test_file_expanded = os.path.expanduser(args.test_file)

    if not os.path.isdir(model_path_expanded):
        logger.error(f"Model directory not found: {model_path_expanded}")
        return
    if not os.path.isfile(test_file_expanded):
        logger.error(f"Test file not found: {test_file_expanded}")
        return

    # --- Load Model and Tokenizer ---
    logger.info(f"Loading model and tokenizer from {model_path_expanded}...")
    try:
        tokenizer = BartTokenizerFast.from_pretrained(model_path_expanded)
        model = BartForConditionalGeneration.from_pretrained(model_path_expanded)
        model.to(args.device)
        model.eval() # Set model to evaluation mode
    except Exception as e:
        logger.error(f"Failed to load model/tokenizer: {e}")
        return

    # --- Load and Tokenize Test Dataset ---
    logger.info(f"Loading and tokenizing test data from {test_file_expanded}...")
    try:
        test_dataset = load_dataset("text", data_files=test_file_expanded)["train"] # Load text file

        def tokenize_function(examples):
            # Tokenize, ensuring truncation and adding special tokens if needed by model
            # For perplexity, we often evaluate on the raw sequence likelihood
            return tokenizer(
                examples["text"],
                truncation=True,
                max_length=args.max_seq_length,
                padding=False, # DataLoader will handle batching, collator handles padding
                return_tensors=None, # Return lists
            )

        tokenized_test_dataset = test_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=os.cpu_count(),
            remove_columns=["text"],
            desc="Tokenizing test set",
        )
    except Exception as e:
        logger.error(f"Failed to load or tokenize test data: {e}")
        return

    # --- Prepare DataLoader ---
    # Use DataCollatorForLanguageModeling to handle padding and potentially mask labels if needed
    # For perplexity, we want the loss on the original sequence, so mlm=False is appropriate
    # Note: BART's denoising objective isn't directly perplexity, but this gives a standard LM measure.
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    test_dataloader = DataLoader(
        tokenized_test_dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator, # Handles padding per batch
    )

    # --- Calculate Perplexity ---
    total_loss = 0
    total_tokens = 0 # Count non-padding tokens for accurate perplexity

    logger.info("Calculating perplexity...")
    progress_bar = tqdm(test_dataloader, desc="Evaluating Perplexity", leave=False)
    with torch.no_grad():
        for batch in progress_bar:
            # Move batch to device
            batch = {k: v.to(args.device) for k, v in batch.items()}

            # Forward pass
            outputs = model(**batch)
            loss = outputs.loss # The model calculates the loss internally

            # Calculate number of non-padding tokens in labels for this batch
            # Labels have -100 for padding, so count where labels are not -100
            num_tokens = (batch["labels"] != -100).sum().item()

            # Accumulate loss weighted by number of tokens
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens

            # Update progress bar if needed
            # progress_bar.set_postfix({'loss': loss.item()})


    if total_tokens == 0:
        logger.error("No tokens were processed. Cannot calculate perplexity.")
        return

    # Calculate average cross-entropy loss
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)

    logger.info(f"Evaluation Complete:")
    logger.info(f"  Average Loss: {avg_loss:.4f}")
    logger.info(f"  Perplexity:   {perplexity:.4f}")
    logger.info(f"  Total non-padding tokens evaluated: {total_tokens}")

if __name__ == "__main__":
    main()
