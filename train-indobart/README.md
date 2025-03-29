# Indonesian BART Pre-training

This project pre-trains a BART model specifically for the Indonesian language, starting from the `facebook/bart-base` checkpoint.

## Setup

1.  **Create Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Steps

1.  **Prepare Data:**
    - Download a large Indonesian text corpus (e.g., OSCAR Indonesian subset).
    - Run `prepare_data.py` to clean and format the data into line-by-line text files suitable for training.
    ```bash
    python prepare_data.py --output_dir ./data --corpus_name oscar --corpus_subset uncommented_deduplicated_id
    ```

2.  **Train Tokenizer:**
    - Run `train_tokenizer.py` to train a SentencePiece or BPE tokenizer on the prepared Indonesian data.
    ```bash
    python train_tokenizer.py --input_files ./data/indonesian.txt --output_dir ./tokenizer --vocab_size 50000
    ```

3.  **Run Pre-training:**
    - Run `run_pretraining.py` using the prepared data and the trained tokenizer.
    - This script will load `facebook/bart-base`, replace its tokenizer embeddings if necessary, and perform BART's denoising pre-training.
    ```bash
    # Example using accelerate for multi-GPU
    accelerate launch run_pretraining.py \
        --model_name_or_path facebook/bart-base \
        --tokenizer_path ./tokenizer \
        --train_file ./data/indonesian.txt \
        --output_dir ./indobart-pretrained \
        --per_device_train_batch_size 8 \
        --gradient_accumulation_steps 4 \
        --learning_rate 5e-5 \
        --num_train_epochs 3 \
        --save_steps 10000 \
        --fp16
    ```

## Notes

- Pre-training requires significant computational resources (GPUs) and time.
- Adjust parameters in the scripts (batch size, learning rate, epochs, etc.) based on your hardware and data size.
