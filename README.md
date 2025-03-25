# IndoBART-v2 Fine-tuning for IndoSUM

This project fine-tunes the IndoBART-v2 model on the IndoSUM dataset for Indonesian text summarization. The trained model is evaluated after each checkpoint and can be pushed to the Hugging Face Hub.

## Setup

### Prerequisites

- Python 3.6+
- PyTorch 1.7+
- Transformers 4.5+
- IndoNLG/indobenchmark toolkit

### Installation

Install the required packages:

```bash
pip install torch transformers pandas numpy tqdm sacrebleu rouge-score indobenchmark-toolkit huggingface_hub
```

## Dataset

The script expects the IndoSUM dataset in HuggingFace Datasets format with the following directory structure:
```
~/dataset/indosum/
├── traindataset/
├── devdataset/
└── testdataset/
```

The dataset should contain at least two columns:
- `document`: The source document to summarize
- `summary`: The target summary

## Training

To train the model with default parameters:

```bash
python train.py
```

### Common Parameters

- `--model_name`: Pretrained model name (default: "indobenchmark/indobart-v2")
- `--output_dir`: Directory to save checkpoints (default: "./checkpoints/indosum")
- `--dataset_dir`: Directory with the IndoSUM dataset (default: "./dataset/IndoSUM")
- `--num_epochs`: Number of training epochs (default: 10)
- `--train_batch_size`: Training batch size (default: 8)
- `--learning_rate`: Initial learning rate (default: 5e-5)
- `--beam_size`: Beam size for generation (default: 5)
- `--fp16`: Use mixed precision training
- `--seed`: Random seed (default: 42)

### Pushing to Hugging Face Hub

To push the fine-tuned model to the Hugging Face Hub:

```bash
python train.py --push_to_hub --hub_model_id "your-username/indobart-v2-indosum" --hub_token "your_hf_token"
```

## Evaluation

The script automatically evaluates the model after each epoch and on the test set after training. Metrics include:
- BLEU
- SacreBLEU
- ROUGE-1, ROUGE-2, ROUGE-L

Results are saved in:
- `validation_metrics.csv`: Metrics for each validation epoch
- `test_metrics.csv`: Final test set metrics
- `test_predictions.csv`: Model predictions vs. gold references

## Outputs

The training script generates:
- Checkpoint files for each epoch
- The best model based on validation performance
- Evaluation metrics for validation and test sets
- Prediction results with model outputs and gold references

## Advanced Configuration

For more configuration options, see the full list of parameters in the script or run:

```bash
python train.py --help
```

## Debugging

If you encounter issues:

1. Check that the dataset files exist and are in the correct format
2. Verify your GPU memory is sufficient for the chosen batch size
3. Try running with a smaller batch size and gradient accumulation
4. Check the IndoNLG library version is compatible with your Transformers version
5. See `plan.md` for more debugging tips
