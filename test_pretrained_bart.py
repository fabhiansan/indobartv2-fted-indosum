import torch
from transformers import BartTokenizerFast, BartForConditionalGeneration
import numpy as np
import random

# --- Configuration ---
PRETRAINED_MODEL_DIR = "./indobart_pretrained" # Directory where the pre-trained model is saved
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# --- Load Model and Tokenizer ---
try:
    print(f"Loading tokenizer from {PRETRAINED_MODEL_DIR}...")
    tokenizer = BartTokenizerFast.from_pretrained(PRETRAINED_MODEL_DIR)
    print(f"Loading model from {PRETRAINED_MODEL_DIR}...")
    model = BartForConditionalGeneration.from_pretrained(PRETRAINED_MODEL_DIR)
    model.to(DEVICE)
    model.eval() # Set model to evaluation mode
    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")
    exit(1)

# --- Text Infilling Test Function ---
def test_text_infilling(text: str, mask_start_ratio: float = 0.3, mask_span_ratio: float = 0.2):
    """
    Tests the BART model's text infilling capability on a given text.

    Args:
        text: The original Indonesian text sentence.
        mask_start_ratio: Approximate starting position for the mask span (as ratio of length).
        mask_span_ratio: Approximate length of the mask span (as ratio of length).
    """
    print("-" * 50)
    print(f"Original Text: {text}")

    # 1. Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt")
    input_ids = inputs["input_ids"][0] # Get the token ids (remove batch dim)

    # Exclude special tokens (BOS/EOS) from masking
    actual_token_ids = input_ids[1:-1] # Assuming BOS is first, EOS is last
    n_tokens = len(actual_token_ids)
    if n_tokens < 5: # Need enough tokens to mask meaningfully
        print("Text too short to apply meaningful mask.")
        return

    # 2. Determine span to mask (simplified for testing)
    span_length = max(1, int(n_tokens * mask_span_ratio))
    start_index = max(0, int(n_tokens * mask_start_ratio))
    end_index = min(n_tokens, start_index + span_length)

    # Adjust if start_index is too close to the end
    if start_index >= n_tokens - 1:
        start_index = max(0, n_tokens - 2)
        end_index = start_index + 1

    # Ensure end_index is valid
    end_index = min(n_tokens, end_index)
    if start_index >= end_index: # Handle edge case where span length becomes 0 or negative
         start_index = max(0, n_tokens - 2)
         end_index = start_index + 1


    # 3. Create the masked input
    masked_token_ids = actual_token_ids[:start_index].tolist() + \
                       [tokenizer.mask_token_id] + \
                       actual_token_ids[end_index:].tolist()

    # Add back BOS and EOS
    masked_input_ids = [tokenizer.bos_token_id] + masked_token_ids + [tokenizer.eos_token_id]
    masked_input_tensor = torch.tensor([masked_input_ids]).to(DEVICE)

    corrupted_text = tokenizer.decode(masked_input_ids, skip_special_tokens=False) # Show mask token
    print(f"Corrupted Text: {corrupted_text}")
    print(f"(Masked span from index {start_index+1} to {end_index}, length {end_index-start_index})") # +1 for BOS

    # 4. Generate the filled text
    with torch.no_grad():
        outputs = model.generate(
            masked_input_tensor,
            max_length=len(input_ids) + 10, # Allow some flexibility in length
            num_beams=4, # Use beam search for better results
            early_stopping=True
        )

    # 5. Decode and print the result
    generated_ids = outputs[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Generated Text: {generated_text}")
    print("-" * 50)


# --- Example Usage ---
sample_texts = [
    "Presiden Joko Widodo mengumumkan ibu kota baru Indonesia akan dipindahkan ke Kalimantan Timur.",
    "Tim nasional sepak bola Indonesia berhasil memenangkan pertandingan melawan Malaysia.",
    "Cuaca di Jakarta hari ini diperkirakan cerah berawan sepanjang hari.",
    "Pemerintah sedang menggalakkan program vaksinasi COVID-19 untuk seluruh masyarakat.",
    "Rendang adalah salah satu makanan terenak di dunia yang berasal dari Sumatera Barat."
]

for sample in sample_texts:
    # Vary mask position slightly for different tests
    test_text_infilling(sample, mask_start_ratio=random.uniform(0.2, 0.6))

print("\nTesting complete.")
