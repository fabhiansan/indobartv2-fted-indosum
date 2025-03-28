"""
Utilities for pushing models to the Hugging Face Hub.
"""
from typing import Dict, List, Optional, Union, Any
import os
import logging
import json
from pathlib import Path

import torch
from transformers import MBartForConditionalGeneration, PreTrainedTokenizer
from transformers.file_utils import TensorType

logger = logging.getLogger(__name__)


def save_model_to_disk(
    model: MBartForConditionalGeneration,
    tokenizer: PreTrainedTokenizer,
    output_dir: str,
    save_tokenizer: bool = True,
    training_args: Optional[Dict[str, Any]] = None,
    eval_results: Optional[Dict[str, Any]] = None
) -> str:
    """
    Save model and tokenizer to disk with enhanced structure and metadata.
    
    Args:
        model: Model to save
        tokenizer: Tokenizer to save
        output_dir: Directory to save to
        save_tokenizer: Whether to save tokenizer
        training_args: Dictionary of training arguments to save as metadata
        eval_results: Dictionary of evaluation results to save as metadata
        
    Returns:
        Path to saved model
    """
    try:
        # Create structured output directories
        model_dir = os.path.join(output_dir, "model")
        tokenizer_dir = os.path.join(output_dir, "tokenizer")
        metadata_dir = os.path.join(output_dir, "metadata")
        
        os.makedirs(model_dir, exist_ok=True)
        if save_tokenizer:
            os.makedirs(tokenizer_dir, exist_ok=True)
        os.makedirs(metadata_dir, exist_ok=True)
        
        logger.info(f"Saving model and tokenizer to structured directory: {output_dir}")
        
        # Save model with additional configuration
        model.save_pretrained(
            model_dir,
            safe_serialization=True
        )
        logger.info(f"Model saved to {model_dir}")
        
        # Save tokenizer with additional configuration
        if save_tokenizer:
            tokenizer.save_pretrained(tokenizer_dir)
            logger.info(f"Tokenizer saved to {tokenizer_dir}")
        
        # Save metadata
        metadata = {
            "model_type": model.config.model_type,
            "model_config": model.config.to_dict(),
            "training_args": training_args or {},
            "eval_results": eval_results or {}
        }
        
        metadata_path = os.path.join(metadata_dir, "metadata.json")
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadata saved to {metadata_path}")
        
        # Verify saved files
        required_files = [
            os.path.join(model_dir, "config.json"),
            os.path.join(model_dir, "pytorch_model.bin")
        ]
        
        if save_tokenizer:
            required_files.extend([
                os.path.join(tokenizer_dir, "tokenizer_config.json"),
                os.path.join(tokenizer_dir, "special_tokens_map.json")
            ])
            
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            raise FileNotFoundError(f"Missing required files after saving: {missing_files}")
        
        logger.info(f"Successfully saved model and tokenizer to {output_dir}")
        logger.info(f"Directory structure:\n{os.listdir(output_dir)}")
        
        return output_dir
    except Exception as e:
        logger.error(f"Failed to save model and tokenizer: {str(e)}")
        logger.exception("Error details:")
        raise RuntimeError(f"Failed to save model: {str(e)}") from e


def push_to_hub(
    model: MBartForConditionalGeneration,
    tokenizer: PreTrainedTokenizer,
    repo_id: str,
    commit_message: str = "Add fine-tuned IndoBART model for IndoSUM",
    private: bool = False,
    token: Optional[str] = None,
    local_dir: Optional[str] = None
) -> None:
    """
    Push model and tokenizer to the Hugging Face Hub.
    
    Args:
        model: Model to push
        tokenizer: Tokenizer to push
        repo_id: Repository ID to push to
        commit_message: Commit message
        private: Whether the repository should be private
        token: Hugging Face token
        local_dir: Local directory to save to before pushing
    """
    logger.info(f"Pushing model and tokenizer to {repo_id}")
    
    # Save model and tokenizer locally if required
    if local_dir is not None:
        save_model_to_disk(model, tokenizer, local_dir, save_tokenizer=True)
    
    # Push model to hub
    model.push_to_hub(
        repo_id=repo_id,
        commit_message=commit_message,
        private=private,
        token=token
    )
    
    # Push tokenizer to hub
    tokenizer.push_to_hub(
        repo_id=repo_id,
        commit_message=commit_message,
        private=private,
        token=token
    )
    
    logger.info(f"Model and tokenizer pushed to {repo_id}")


def create_model_card(
    repo_id: str,
    metrics: Dict[str, float],
    model_name: str = "IndoBART-v2",
    dataset_name: str = "IndoSUM",
    language: str = "Indonesian",
    license_name: str = "MIT",
    finetuned_from: str = "indobenchmark/indobart",
    tasks: List[str] = ["summarization"],
    output_file: str = "README.md",
    training_data: str = "Fine-tuned on the IndoSUM dataset",
    training_procedure: str = "Fine-tuning with a denoising auto-encoding objective",
    intended_use: str = "Indonesian text summarization"
) -> str:
    """
    Create a model card for the Hugging Face Hub.

    Args:
        repo_id: Repository ID
        metrics: Dictionary of metrics
        model_name: Name of the model
        dataset_name: Name of the dataset
        language: Language of the model
        license_name: License name
        finetuned_from: Model this was fine-tuned from
        tasks: List of tasks the model performs
        output_file: Output file to save to
        training_data: Description of the training data
        training_procedure: Description of the training procedure
        intended_use: Description of the intended use
        
    Returns:
        Markdown string of the model card
    """
    # Format metrics for the model card
    metrics_table = "| Metric | Value |\n| ------ | ----- |\n"
    for name, value in metrics.items():
        metrics_table += f"| {name} | {value:.2f} |\n"

    # Create citation with URL
    citation = f"""@misc{{indobart-indosum,
  author = {{{{IndoNLP Team}}}},
  title = {{{{IndoBART fine-tuned on IndoSUM}}}},
  year = {{2025}},
  publisher = {{Hugging Face}},
  howpublished = {{\\\\url{{https://huggingface.co/{repo_id}}}}}
}}"""

    # Create model card content
    model_card = f"""---
language: {language.lower()}
license: {license_name}
tags:
- {model_name.lower()}
- {language.lower()}
- summarization
- indobart
datasets:
- {dataset_name.lower()}
metrics:
- rouge
- bleu
- bertscore
model-index:
- name: {model_name} fine-tuned on {dataset_name}
  results:
  - task:
      type: summarization
      name: Summarization
    dataset:
      name: {dataset_name}
      type: {dataset_name.lower()}
    metrics:
{metrics_table.replace('|', '      -')}
---

# {model_name} Fine-tuned for {language} Summarization

This model is a fine-tuned version of [{finetuned_from}](https://huggingface.co/{finetuned_from}) on the {dataset_name} dataset.
It achieves the following results on the evaluation set:

{metrics_table}

## Model Description

This model was {training_data} for {language} text summarization. It's based on the {model_name} model, which is a {language} language model pre-trained with a denoising auto-encoding objective. The model was fine-tuned using {training_procedure}.

## Intended Use and Limitations

This model is intended to be used for {intended_use}. It may not perform well on other languages or tasks.

## Training and Evaluation Data

The model was trained on the {dataset_name} dataset, which consists of {language} documents and their summaries.

## Training Procedure

The model was fine-tuned on the {dataset_name} training set and evaluated on the validation and test sets.

## How to Use

```python
from transformers import MBartForConditionalGeneration, AutoTokenizer

# Load the model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("{repo_id}")
tokenizer = AutoTokenizer.from_pretrained("{repo_id}")

# Prepare the text to be summarized
text = "YOUR_TEXT_TO_SUMMARIZE"

# Tokenize the text
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)

# Generate the summary
summary_ids = model.generate(
    inputs["input_ids"],
    num_beams=4,
    max_length=150,
    early_stopping=True,
    no_repeat_ngram_size=3
)

# Decode the summary
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
print(summary)
```

## Limitations and Biases

* This model is limited by its training data and may not perform well on text that differs significantly from the training data.
* The model may reproduce biases present in the training data.

## Citation

```
{citation}
```
"""

    # Write model card to file if requested
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(model_card)

    return model_card
