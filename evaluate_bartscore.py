# evaluate_bartscore.py

import torch
from bart_score import BARTScorer
import logging
import sys
import finetune_config as cfg # Import config to get the output directory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
# Path to the directory where your fine-tuned model was saved
# This should match the OUTPUT_DIR from finetune_config.py
FINETUNED_MODEL_PATH = cfg.OUTPUT_DIR

# Use CUDA if available, otherwise fallback to CPU
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4 # Adjust based on your GPU memory, if using GPU

# --- Input Data ---
# Replace these with the actual document and summary you want to evaluate
# Option 1: Single document and summary
# documents = ["Ini adalah dokumen sumber yang panjang tentang peristiwa penting di Jakarta."]
# summaries = ["Peristiwa penting terjadi di Jakarta."]

# Option 2: Multiple documents and summaries (must be lists of the same length)
documents = [
    "Pemerintah mengumumkan rencana baru untuk infrastruktur transportasi di ibu kota. Proyek ini diharapkan mengurangi kemacetan secara signifikan.",
    "Tim bulu tangkis Indonesia berhasil meraih medali emas di kejuaraan internasional setelah mengalahkan lawan tangguh di final."
]
summaries = [
    "Pemerintah umumkan rencana infrastruktur baru.", # Candidate summary 1
    "Indonesia raih emas bulu tangkis." # Candidate summary 2
]

# --- Script Logic ---
print(f"--- BARTScore Evaluation Script ---")
print(f"Using Fine-tuned Model: {FINETUNED_MODEL_PATH}")
print(f"Using Device: {DEVICE}")
print("-" * 30)

# Input validation
if not isinstance(documents, list) or not isinstance(summaries, list):
    print("Error: 'documents' and 'summaries' must be lists of strings.")
    sys.exit(1)
if len(documents) != len(summaries):
    print(f"Error: Number of documents ({len(documents)}) must match number of summaries ({len(summaries)}).")
    sys.exit(1)
if not documents:
    print("Error: Input lists cannot be empty.")
    sys.exit(1)

try:
    # 1. Initialize BARTScorer with the fine-tuned model
    print("Loading BARTScorer with fine-tuned model...")
    # Ensure the path exists (basic check)
    import os
    if not os.path.isdir(FINETUNED_MODEL_PATH):
        print(f"Error: Fine-tuned model directory not found at '{FINETUNED_MODEL_PATH}'")
        print("Please ensure the training completed successfully and the path is correct.")
        sys.exit(1)

    bart_scorer = BARTScorer(device=DEVICE, checkpoint=FINETUNED_MODEL_PATH)
    print("BARTScorer loaded successfully.")
    print("-" * 30)

    # 2. Calculate Scores
    # BARTScore typically evaluates the summary based on the document (summary | document)
    # Or the faithfulness of the summary to the document (document | summary)
    # The original BARTScore paper uses P(summary | document) for quality.
    # Let's calculate P(summary | document) - higher score means the model thinks
    # the summary is a likely generation given the document.
    print("Calculating BARTScore (summary | document)...")
    scores = bart_scorer.score(
        documents, # Source texts (conditioned on)
        summaries, # Target texts (whose probability is calculated)
        batch_size=BATCH_SIZE
    )
    print("Scores calculated.")
    print("-" * 30)

    # 3. Print Results
    print("Results (Higher score is better):")
    for i, score in enumerate(scores):
        print(f"Example {i+1}:")
        print(f"  Document: \"{documents[i][:100]}...\"") # Print truncated document
        print(f"  Summary:  \"{summaries[i]}\"")
        print(f"  BARTScore (summary | document): {score:.4f}")
        print("-" * 10)

except ImportError:
     print("Error: 'bart_score' library not found. Please install it: pip install bart_score")
except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please check the model path and ensure libraries are installed correctly.")
    if "CUDA" in str(e):
        print("If using GPU, ensure CUDA is set up correctly.")

print("\n--- Evaluation Script Finished ---")