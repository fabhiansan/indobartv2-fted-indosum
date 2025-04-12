# bartscore_calculator.py

import torch
from bart_score import BARTScorer
import logging
import sys
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default configuration (can be overridden when calling the function)
DEFAULT_DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
DEFAULT_BATCH_SIZE = 8

def calculate_bartscore(model_path, documents, summaries, device=DEFAULT_DEVICE, batch_size=DEFAULT_BATCH_SIZE):
    """
    Loads a BART-based model and calculates BARTScore (P(summary | document)).

    Args:
        model_path (str): Path to the pre-trained or fine-tuned model directory
                          (e.g., './indobart-finetuned-summarization' or 'facebook/bart-large-cnn').
        documents (list[str]): A list of source documents.
        summaries (list[str]): A list of corresponding summaries.
        device (str, optional): Device to run on ('cuda:0', 'cpu', etc.). Defaults to DEFAULT_DEVICE.
        batch_size (int, optional): Batch size for scoring. Defaults to DEFAULT_BATCH_SIZE.

    Returns:
        list[float]: A list of BARTScore values for each document-summary pair,
                     or None if an error occurs.
    """
    logging.info(f"Calculating BARTScore using model: {model_path} on device: {device}")

    # Input validation
    if not isinstance(documents, list) or not isinstance(summaries, list):
        logging.error("Input 'documents' and 'summaries' must be lists of strings.")
        return None
    if len(documents) != len(summaries):
        logging.error(f"Number of documents ({len(documents)}) must match number of summaries ({len(summaries)}).")
        return None
    if not documents:
        logging.error("Input lists cannot be empty.")
        return None
    if not os.path.isdir(model_path):
         # Check if it's potentially a Hugging Face Hub identifier (basic check)
         if "/" not in model_path:
              logging.error(f"Model path '{model_path}' is not a valid directory.")
              return None
         else:
              logging.info(f"'{model_path}' is not a local directory, assuming it's a Hugging Face Hub identifier.")


    try:
        # Initialize BARTScorer
        logging.info("Loading BARTScorer...")
        bart_scorer = BARTScorer(device=device, checkpoint=model_path)
        logging.info("BARTScorer loaded.")

        # Calculate Scores P(summary | document)
        logging.info("Calculating scores...")
        scores = bart_scorer.score(
            documents, # Source texts (conditioned on)
            summaries, # Target texts (whose probability is calculated)
            batch_size=batch_size
        )
        logging.info("Scores calculated.")
        return scores

    except ImportError:
         logging.error("'bart_score' library not found. Please install it: pip install bart_score")
         return None
    except Exception as e:
        logging.error(f"An error occurred during BARTScore calculation: {e}", exc_info=True)
        return None

# Example usage if the script is run directly
if __name__ == '__main__':
    print("--- Running BARTScore Calculator Example ---")

    # --- Example Configuration ---
    # Use the fine-tuned model path from your config, or a standard one like 'facebook/bart-large-cnn'
    try:
        import finetune_config as cfg
        MODEL_TO_TEST = cfg.OUTPUT_DIR # Assumes finetune_config.py and output dir exist
        print(f"Using fine-tuned model path from config: {MODEL_TO_TEST}")
    except (ImportError, AttributeError):
        MODEL_TO_TEST = "facebook/bart-large-cnn" # Fallback to a standard model
        print(f"Fine-tune config not found or OUTPUT_DIR not set. Using fallback model: {MODEL_TO_TEST}")


    # Example data
    example_docs = [
        "Pemerintah mengumumkan rencana baru untuk infrastruktur transportasi di ibu kota. Proyek ini diharapkan mengurangi kemacetan secara signifikan.",
        "Tim bulu tangkis Indonesia berhasil meraih medali emas di kejuaraan internasional setelah mengalahkan lawan tangguh di final."
    ]
    example_sums = [
        "Pemerintah umumkan rencana infrastruktur baru.",
        "Indonesia raih emas bulu tangkis."
    ]

    # Calculate scores using the function
    calculated_scores = calculate_bartscore(MODEL_TO_TEST, example_docs, example_sums)

    # Print results
    if calculated_scores is not None:
        print("\n--- Example Results ---")
        for i, score in enumerate(calculated_scores):
            print(f"Pair {i+1}:")
            print(f"  Document: \"{example_docs[i][:50]}...\"")
            print(f"  Summary:  \"{example_sums[i]}\"")
            print(f"  BARTScore: {score:.4f}")
    else:
        print("\nBARTScore calculation failed. See logs above for details.")

    print("\n--- Example Finished ---")