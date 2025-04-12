# use_bartscore_calculator.py

# Import the function from the other script
from bartscore_calculator import calculate_bartscore
import finetune_config as cfg # To get the output directory easily

# --- Configuration ---
# Specify the model path you want to use for scoring
# Option 1: Use the fine-tuned model saved by the training script
MODEL_PATH = cfg.OUTPUT_DIR
# Option 2: Use a different pre-trained model from Hugging Face Hub
# MODEL_PATH = "facebook/bart-large-cnn"
# Option 3: Use the base IndoBART model we started with
# MODEL_PATH = "gaduhhartawan/indobart-base-v2"

# --- Your Data ---
my_documents = [
    "Artikel panjang tentang pertandingan sepak bola tadi malam antara tim A dan tim B.",
    "Laporan cuaca memprediksi hujan lebat akan turun di wilayah Jabodetabek sore ini.",
    "Debat kandidat presiden berlangsung sengit membahas isu ekonomi dan sosial."
]
my_summaries = [
    "Tim A menang 2-1 atas tim B.",
    "Jabodetabek diprediksi hujan lebat.",
    "Debat capres bahas ekonomi."
]

print(f"--- Using BARTScore Calculator ---")
print(f"Model for scoring: {MODEL_PATH}")

# --- Calculate Scores ---
bart_scores = calculate_bartscore(
    model_path=MODEL_PATH,
    documents=my_documents,
    summaries=my_summaries
    # Optionally override device or batch_size:
    # device='cpu',
    # batch_size=2
)

# --- Process Results ---
if bart_scores is not None:
    print("\n--- Calculated Scores ---")
    for i, score in enumerate(bart_scores):
        print(f"Pair {i+1}:")
        print(f"  Document: \"{my_documents[i][:50]}...\"")
        print(f"  Summary:  \"{my_summaries[i]}\"")
        print(f"  BARTScore: {score:.4f}")
else:
    print("\nFailed to calculate BARTScore. Check logs.")

print("\n--- Finished ---")