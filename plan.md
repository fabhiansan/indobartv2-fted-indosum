# Plan: Using BARTScore with IndoBART

This plan outlines the steps to evaluate generated text using BARTScore with a custom IndoBART model. It assumes IndoBART is compatible with the Hugging Face Transformers library.

1.  **Prerequisites:**
    *   Ensure Python environment is set up.
    *   Install required libraries:
        ```bash
        pip install -r BARTScore/requirements.txt
        ```
    *   Have access to the IndoBART model (either as a Hugging Face Hub identifier or a local path).

2.  **Import BARTScorer:**
    *   In your Python script, import the necessary class:
        ```python
        from bart_score import BARTScorer
        ```

3.  **Initialize BARTScorer with IndoBART:**
    *   Create an instance of `BARTScorer`.
    *   Set the `checkpoint` parameter to the identifier or path of your IndoBART model.
    *   Specify the `device` (e.g., `'cuda:0'` for GPU or `'cpu'`).
        ```python
        # --- Configuration ---
        INDOBART_MODEL_PATH = 'gaduhhartawan/indobart-base-v2' # BART-based model fine-tuned on id_liputan6 (summarization)
        DEVICE = 'cuda:0' # Or 'cpu'
        BATCH_SIZE = 4 # Adjust based on your GPU memory
        # ---------------------

        print(f"Loading BARTScorer with model: {INDOBART_MODEL_PATH}")
        bart_scorer = BARTScorer(device=DEVICE, checkpoint=INDOBART_MODEL_PATH)
        print("BARTScorer loaded.")
        ```

4.  **Prepare Data:**
    *   Create two lists of strings:
        *   `candidates`: The list of generated texts (hypotheses) you want to evaluate.
        *   `references`: The list of corresponding ground-truth texts.
        *   *Optional:* For multiple references per candidate, prepare `references` as a list of lists.
    ```python
    # Example Data (replace with your actual data)
    candidates = [
        "Ini adalah teks yang dihasilkan oleh sistem.",
        "Contoh lain dari teks kandidat."
    ]
    references = [
        "Ini adalah teks referensi yang benar.",
        "Contoh teks acuan lainnya."
    ]

    # Example for multi-reference
    # multi_references = [
    #     ["Referensi pertama untuk kandidat 1.", "Referensi alternatif untuk kandidat 1."],
    #     ["Referensi tunggal untuk kandidat 2."]
    # ]
    ```

5.  **Calculate Scores:**
    *   Use the `score()` method for single references or `multi_ref_score()` for multiple references.
    *   Remember that scores are log-likelihoods (negative values), where higher values indicate better scores.
    ```python
    print("Calculating BARTScore...")
    # For single reference per candidate
    scores = bart_scorer.score(candidates, references, batch_size=BATCH_SIZE)

    # For multiple references per candidate (choose aggregation method: 'max' or 'mean')
    # scores = bart_scorer.multi_ref_score(candidates, multi_references, agg="max", batch_size=BATCH_SIZE)

    print("Scores calculated:")
    for i, score in enumerate(scores):
        print(f"Candidate {i+1}: {score:.4f}")

    # Example interpretation:
    # If score A = -1.5 and score B = -3.0, then candidate A is considered better by IndoBART.
    ```

6.  **Next Steps:**
    *   Integrate this scoring logic into your evaluation pipeline.
    *   Analyze the correlation of these scores with human judgments if available.