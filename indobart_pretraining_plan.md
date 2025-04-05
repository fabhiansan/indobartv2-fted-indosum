# Plan Checklist: Pre-training Indonesian BART from Scratch (Script Generation)

**Overall Goal:** Generate Python and bash scripts to pre-train a BART-based model for Indonesian using Hugging Face tools, suitable for execution on a remote server (e.g., Jupyter Hub). The primary downstream task focus is text summarization.

---

## Step 1: Dataset Acquisition & Preparation

*   **Objective:** Create scripts to load, clean, and format the Indonesian OSCAR corpus.
*   **Target Dataset:** Indonesian subset of the **OSCAR corpus** (cached locally).
*   **Actions:**
    - [x] **Create Data Loading/Cleaning Script (`preprocess_data.py`):**
        - [x] Load dataset from cache (`~/.cache/huggingface/datasets/oscar-corpus___oscar-2301/id`).
        - [x] Implement cleaning functions (deduplication, quality filtering, normalization).
        - [x] Apply cleaning to the dataset.
        - [x] Save the cleaned data as plain text files (e.g., `cleaned_oscar_id.txt`).
    - [x] **Create Bash Script (`run_preprocessing.sh`):**
        - [x] Script to execute `preprocess_data.py` with necessary arguments (e.g., input cache path, output file path).
*   **Considerations:** Cleaning requires careful implementation; processing time can still be significant.

---

## Step 2: Tokenizer Training

*   **Objective:** Create scripts to train a custom BPE tokenizer.
*   **Algorithm:** Byte-Pair Encoding (BPE) using `tokenizers.ByteLevelBPETokenizer`.
*   **Input:** Cleaned Indonesian text file(s) from Step 1 (`cleaned_oscar_id.txt`).
*   **Actions:**
    - [x] **Create Tokenizer Training Script (`train_tokenizer.py`):**
        - [x] Define special tokens: `<s>`, `</s>`, `<pad>`, `<unk>`, `[MASK]`.
        - [x] Instantiate `ByteLevelBPETokenizer`.
        - [x] Implement training logic using `tokenizer.train()`.
        - [x] Include argument parsing for input file(s), output directory, vocab size, min frequency.
        - [x] Save the trained tokenizer files (`vocab.json`, `merges.txt`) to the specified output directory.
    - [x] **Create Bash Script (`run_tokenizer_training.sh`):**
        - [x] Script to execute `train_tokenizer.py` with arguments (e.g., input text file path, output tokenizer path, vocab size).
*   **Integration:** The pre-training script will load the tokenizer from the output directory.

---

## Step 3: Model Configuration

*   **Objective:** Define the BART model architecture *within* the pre-training script.
*   **Architecture:** **BART-base** (6 encoder/decoder layers, 768 hidden size, 12 heads).
*   **Actions:**
    - [x] *(No separate script needed)* Configuration will be handled inside `run_pretraining.py` (Step 4) using `transformers.BartConfig`.
    - [x] Ensure `vocab_size` in the config matches the trained tokenizer's size.
    - [x] Ensure special token IDs are correctly mapped.

---

## Step 4: Pre-training Script & Data Collator

*   **Objective:** Create the main Python script for pre-training the model.
*   **Framework:** PyTorch, `transformers`, `datasets`, `tokenizers`.
*   **Training API:** Hugging Face `Trainer`.
*   **Actions:**
    - [x] **Develop Pre-training Script (`run_pretraining.py`):**
        - [x] Implement argument parsing (model output path, tokenizer path, dataset path, training args).
        - [x] Load custom tokenizer (`BartTokenizerFast.from_pretrained`, from Step 2 output).
        - [x] Define/Load model config (`BartConfig`) ensuring `vocab_size` matches tokenizer.
        - [x] Instantiate model (`BartForConditionalGeneration.from_config` - *no pre-trained weights*).
        - [x] Load the *tokenized* dataset (requires a step to tokenize the cleaned text from Step 1 using the Step 2 tokenizer, potentially done within this script or a separate one).
        - [x] Implement the custom `DataCollatorForBartDenoising` (Text Infilling logic) within the script or as a separate importable module.
        - [x] Define `TrainingArguments`.
        - [x] Initialize `Trainer`.
        - [x] Include logic to call `trainer.train()` and `trainer.save_model()`.
    - [x] **Create Bash Script (`run_pretraining.sh`):**
        - [x] Script to execute `run_pretraining.py` using `torchrun` or `accelerate launch` for potential distributed training.
        - [x] Pass all necessary arguments (paths, batch size, LR, epochs, GPU settings, etc.).

---

## Step 5: Pre-training Execution

*   **Objective:** Provide the means to run the pre-training process.
*   **Actions:**
    - [x] *(Covered by `run_pretraining.sh`)* The user will execute this script on their server.
    - [x] **Add Notes on Environment Setup:** Include comments in the bash script or plan about required libraries (PyTorch, transformers, datasets, tokenizers, accelerate).
    - [x] **Add Notes on Monitoring:** Include comments about using logging tools (TensorBoard/W&B) configured via `TrainingArguments`.

---

## Step 6: Save & Evaluate

*   **Objective:** Ensure the model is saved correctly; provide optional evaluation script.
*   **Actions:**
    - [x] *(Saving handled by `run_pretraining.py`)* The `Trainer` saves the final model/tokenizer.
    - [x] *(Optional)* **Create Evaluation Script (`evaluate_perplexity.py`):**
        - [x] Script to load the trained model/tokenizer.
        - [x] Load a held-out test set.
        - [x] Calculate and report perplexity.
    - [x] *(Optional)* **Create Bash Script (`run_evaluation.sh`):**
        - [x] Script to execute `evaluate_perplexity.py`.
*   **Output:** A directory containing the pre-trained model/tokenizer.

---

## Notes:

*   **CUDA Error Mitigation:** Consistency between tokenizer `vocab_size` (Step 2) and model config `vocab_size` (Step 4 script) remains crucial.
*   **IndoBART-v2:** This plan builds a new model using standard tools.
