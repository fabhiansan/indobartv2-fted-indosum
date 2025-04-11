# test_bartscore.py

import torch
from bart_score import BARTScorer

# --- Configuration ---
# Model fine-tuned on Indonesian summarization (id_liputan6 dataset)
MODEL_CHECKPOINT = 'gaduhhartawan/indobart-base-v2'
# Use CUDA if available, otherwise fallback to CPU
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 4 # Adjust based on your GPU memory, if using GPU

# --- Example Data ---
# Replace these with your actual candidate and reference summaries
candidate_summaries = [
    "Presiden mengunjungi korban bencana alam di Jawa Timur hari ini.", # Candidate summary 1
    "Pemerintah akan menaikkan harga bahan bakar minyak dalam waktu dekat.", # Candidate summary 2
    "Timnas sepak bola berhasil memenangkan pertandingan persahabatan." # Candidate summary 3
]

reference_summaries = [
    "Presiden Joko Widodo hari ini melakukan kunjungan kerja ke Jawa Timur untuk meninjau langsung kondisi para korban bencana alam.", # Reference summary 1
    "Kenaikan harga bahan bakar minyak (BBM) akan segera diumumkan oleh pemerintah sebagai bagian dari penyesuaian anggaran.", # Reference summary 2
    "Tim nasional sepak bola Indonesia meraih kemenangan dalam laga uji coba internasional melawan tim tamu." # Reference summary 3
]

print(f"--- BARTScore Test Script ---")
print(f"Using Model: {MODEL_CHECKPOINT}")
print(f"Using Device: {DEVICE}")
print(f"Batch Size: {BATCH_SIZE}")
print("-" * 30)

try:
    # 1. Initialize BARTScorer
    print("Loading BARTScorer...")
    # Note: The BARTScorer might download the model files if not already cached.
    bart_scorer = BARTScorer(device=DEVICE, checkpoint=MODEL_CHECKPOINT)
    print("BARTScorer loaded successfully.")
    print("-" * 30)

    # 2. Calculate Scores
    print("Calculating BARTScore for example summaries...")
    # The score method takes a list of candidates and a list of references
    scores = bart_scorer.score(
        candidate_summaries,
        reference_summaries,
        batch_size=BATCH_SIZE
    )
    print("Scores calculated.")
    print("-" * 30)

    # 3. Print Results
    print("Results (Higher score is better):")
    for i, score in enumerate(scores):
        print(f"Example {i+1}:")
        print(f"  Candidate: \"{candidate_summaries[i]}\"")
        print(f"  Reference: \"{reference_summaries[i]}\"")
        print(f"  BARTScore: {score:.4f}")
        print("-" * 10)

except Exception as e:
    print(f"\nAn error occurred: {e}")
    print("Please ensure you have the 'bart_score' library installed (`pip install bart_score`)")
    print("Also check your internet connection if the model needs to be downloaded.")
    print("If using GPU, ensure CUDA is set up correctly.")

print("\n--- Test Script Finished ---")