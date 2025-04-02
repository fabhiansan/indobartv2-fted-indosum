# IndoBART Pre-training and Fine-tuning Plan

This document outlines the steps to adapt a BART base model for Indonesian and fine-tune it for summarization.

**Goal:** Create an Indonesian BART model capable of text summarization.

**Approach:** Option A (Continue Pre-training + Fine-tuning) using the OSCAR corpus and a custom Indonesian tokenizer.

**Steps:**

1.  **Data Acquisition:**
    *   **Large Corpus:** Use the Indonesian subset of the OSCAR corpus (e.g., `oscar-corpus/OSCAR-2301`, language `id`) accessed via the Hugging Face `datasets` library. Requires a Hugging Face account and agreeing to terms.
    *   **Summarization Data:** Obtain a suitable Indonesian summarization dataset (e.g., IndoSum, Liputan6, etc.). This needs to be in a format with 'document' and 'summary' columns/fields.

2.  **Environment Setup:**
    *   Install necessary Python libraries: `torch`, `transformers`, `datasets`, `tokenizers`, `sentencepiece` (or other tokenizer library).
    *   Ensure access to a suitable GPU environment (e.g., local GPU with CUDA drivers, cloud instance, Jupyter Hub with GPU).

3.  **Tokenizer Training (`train_tokenizer.py`):**
    *   Load the OSCAR Indonesian dataset using `datasets` (potentially streaming).
    *   Prepare the text data (e.g., write to temporary files if needed by the tokenizer trainer).
    *   Train a new tokenizer (e.g., SentencePiece BPE) on the OSCAR data using the `tokenizers` library.
    *   Specify vocabulary size (e.g., 30k-50k).
    *   Save the trained tokenizer files (e.g., `vocab.json`, `merges.txt` or `tokenizer.model`).

4.  **Continued Pre-training (`run_pretraining.py`):**
    *   **Load Base Model:** Load the `facebook/bart-base` model using `transformers`.
    *   **Load New Tokenizer:** Load the custom Indonesian tokenizer trained in the previous step.
    *   **Replace Tokenizer & Resize Embeddings:** Assign the new tokenizer to the model and resize the model's token embeddings to match the new vocabulary size (`model.resize_token_embeddings(len(new_tokenizer))`).
    *   **Prepare Pre-training Data:** Use the OSCAR dataset again. Implement BART's denoising objective (text infilling, sentence permutation) using a data collator (e.g., `DataCollatorForLanguageModeling` adapted for BART or a custom one).
    *   **Configure Training:** Set up `TrainingArguments` (output directory, learning rate, batch size, number of epochs/steps, gradient accumulation, etc.). Use parameters suitable for large-scale pre-training.
    *   **Initialize Trainer:** Create a `Trainer` instance with the model, training arguments, datasets, data collator, and the new tokenizer.
    *   **Run Training:** Execute `trainer.train()`. This step is computationally intensive.
    *   **Save Model:** Save the pre-trained Indonesian BART model and tokenizer using `trainer.save_model()`.

5.  **Fine-tuning for Summarization (`run_summarization.py`):**
    *   **Load Pre-trained Model:** Load the Indonesian BART model and tokenizer saved from the previous step.
    *   **Load Summarization Data:** Load the Indonesian summarization dataset (e.g., IndoSum) using `datasets`. Ensure it has 'document' and 'summary' columns.
    *   **Preprocess Data:** Create a function to tokenize the documents (input) and summaries (labels) using the loaded tokenizer. Handle padding and truncation. Map this function over the dataset.
    *   **Data Collator:** Use `DataCollatorForSeq2Seq`.
    *   **Configure Training:** Set up `Seq2SeqTrainingArguments` (output directory, learning rate, batch size, epochs, evaluation strategy, generation parameters like `max_length`, `num_beams`, etc.).
    *   **Metrics:** Define a function to compute ROUGE scores using the `evaluate` (formerly `datasets`) library.
    *   **Initialize Trainer:** Create a `Seq2SeqTrainer` instance with the model, training arguments, datasets, data collator, tokenizer, and compute_metrics function.
    *   **Run Training:** Execute `trainer.train()`.
    *   **Evaluate:** Run `trainer.evaluate()` on the test set.
    *   **Save Model:** Save the final fine-tuned summarization model and tokenizer using `trainer.save_model()`.

6.  **Usage:** Load the final model and tokenizer for inference to summarize new Indonesian text.
