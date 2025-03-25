# IndoBART Fine-tuning for IndoSUM Dataset

This repository contains code for fine-tuning the IndoBART-v2 model on the IndoSUM dataset for Indonesian text summarization.

## Project Structure

- `main.py`: Main script that orchestrates the entire fine-tuning process
- `data_loader.py`: Dataset handling and data loading utilities
- `train.py`: Training functionality and loop implementation
- `evaluate.py`: Evaluation metrics and prediction functionality
- `hub_utils.py`: Utilities for pushing models to Hugging Face Hub
- `plan.md`: Project plan and progress tracking

## Requirements

```
torch>=1.7.0
transformers>=4.12.0
datasets>=1.10.0
rouge_score>=0.0.4
sacrebleu>=1.5.0
nltk>=3.6.0
pandas>=1.1.0
tqdm>=4.50.0
```

Install the requirements using:

```bash
pip install -r requirements.txt
```

## Dataset

The IndoSUM dataset should be organized in the following structure:

```
../dataset/indosum/
├── traindataset/
├── devdataset/
└── testdataset/
```

Each directory should contain the dataset in the Hugging Face Datasets format, with at least "document" and "summary" columns.

## Usage

### Training

To fine-tune the IndoBART-v2 model on the IndoSUM dataset:

```bash
python main.py \
    --data_dir ../dataset/indosum \
    --output_dir ./results \
    --model_dir ./save \
    --batch_size 8 \
    --lr 3e-5 \
    --n_epochs 5 \
    --evaluate_every 1 \
    --early_stop 3 \
    --valid_criterion ROUGE1 \
    --beam_size 5
```

### Options

- `--data_dir`: Directory containing the dataset (default: ../dataset/indosum)
- `--output_dir`: Directory to save outputs (default: ./results)
- `--model_dir`: Directory to save models (default: ./save)
- `--model_name`: Pretrained model name or path (default: indobenchmark/indobart)
- `--max_seq_len`: Maximum sequence length (default: 512)
- `--max_source_length`: Maximum source sequence length (default: 1024)
- `--max_target_length`: Maximum target sequence length (default: 256)
- `--batch_size`: Batch size (default: 8)
- `--lr`: Learning rate (default: 3e-5)
- `--n_epochs`: Number of epochs (default: 5)
- `--evaluate_every`: Evaluate every n epochs (default: 1)
- `--early_stop`: Early stopping patience (default: 3)
- `--valid_criterion`: Validation criterion (default: ROUGE1)
- `--beam_size`: Beam size for generation (default: 5)
- `--fp16`: Use FP16 precision (flag)
- `--push_to_hub`: Push model to Hugging Face Hub (flag)
- `--hub_model_id`: Repository ID for Hugging Face Hub
- `--hub_private`: Make Hugging Face Hub repository private (flag)

## Pushing to Hugging Face Hub

To fine-tune and push the model to Hugging Face Hub:

```bash
python main.py \
    --push_to_hub \
    --hub_model_id your-username/indobart-finetuned-indosum \
    --hub_token your_token
```

You'll need to set up your Hugging Face credentials first.

## Using the Fine-tuned Model

After fine-tuning, you can use the model for summarization:

```python
from transformers import MBartForConditionalGeneration, AutoTokenizer

# Load the model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("path/to/model")
tokenizer = AutoTokenizer.from_pretrained("path/to/model")

# Prepare the text to be summarized
text = "YOUR_INDONESIAN_TEXT_TO_SUMMARIZE"

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)

# Generate the summary
summary_ids = model.generate(
    inputs["input_ids"],
    num_beams=5,
    max_length=256,
    early_stopping=True,
    no_repeat_ngram_size=3,
    length_penalty=1.0
)

# Decode the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
```

## Evaluation Metrics

The model is evaluated using the following metrics:
- BLEU
- SacreBLEU
- ROUGE1, ROUGE2, ROUGEL, ROUGELsum

## References

- [IndoBART](https://huggingface.co/indobenchmark/indobart)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
