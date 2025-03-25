#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Utility functions for IndoBART-v2 fine-tuning on IndoSUM dataset.
"""

import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import torch
from transformers import MBartForConditionalGeneration
from huggingface_hub import HfApi, HfFolder

logger = logging.getLogger(__name__)

def plot_training_metrics(metrics_file: str, output_dir: str) -> None:
    """
    Plot training metrics from CSV file.
    
    Args:
        metrics_file: Path to metrics CSV file
        output_dir: Directory to save plots
    """
    if not os.path.exists(metrics_file):
        logger.error(f"Metrics file {metrics_file} not found.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load metrics
    metrics_df = pd.read_csv(metrics_file)
    
    # Get list of metrics
    metric_columns = metrics_df.columns
    
    # Plot each metric
    for metric in metric_columns:
        plt.figure(figsize=(10, 6))
        plt.plot(metrics_df.index + 1, metrics_df[metric], marker='o')
        plt.title(f'{metric} over epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{metric}_plot.png'))
        plt.close()
    
    # Plot all metrics together
    plt.figure(figsize=(12, 8))
    for metric in metric_columns:
        plt.plot(metrics_df.index + 1, metrics_df[metric], marker='o', label=metric)
    plt.title('Training Metrics over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'all_metrics_plot.png'))
    plt.close()

def analyze_predictions(predictions_file: str, output_dir: str, n_samples: int = 10) -> None:
    """
    Analyze model predictions and generate report.
    
    Args:
        predictions_file: Path to predictions CSV file
        output_dir: Directory to save analysis
        n_samples: Number of example samples to include in report
    """
    if not os.path.exists(predictions_file):
        logger.error(f"Predictions file {predictions_file} not found.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load predictions
    pred_df = pd.read_csv(predictions_file)
    
    # Calculate summary statistics
    summary = {
        'total_samples': len(pred_df),
        'avg_hyp_length': pred_df['hyp'].apply(lambda x: len(x.split())).mean(),
        'avg_label_length': pred_df['label'].apply(lambda x: len(x.split())).mean(),
        'exact_matches': (pred_df['hyp'] == pred_df['label']).sum(),
    }
    
    # Calculate length difference
    pred_df['length_diff'] = pred_df['hyp'].apply(len) - pred_df['label'].apply(len)
    
    # Categorize samples
    pred_df['category'] = pd.cut(
        pred_df['length_diff'],
        bins=[-float('inf'), -20, -5, 5, 20, float('inf')],
        labels=['much_shorter', 'shorter', 'similar', 'longer', 'much_longer']
    )
    
    # Count categories
    category_counts = pred_df['category'].value_counts().to_dict()
    
    # Add to summary
    summary['length_categories'] = category_counts
    
    # Get example samples from each category
    examples = {}
    for category in pred_df['category'].unique():
        category_df = pred_df[pred_df['category'] == category]
        if len(category_df) > 0:
            sample_count = min(n_samples, len(category_df))
            examples[category] = category_df.sample(sample_count)[['hyp', 'label']].to_dict('records')
    
    # Create full report
    report = {
        'summary': summary,
        'examples': examples
    }
    
    # Save report as JSON
    with open(os.path.join(output_dir, 'prediction_analysis.json'), 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create a readable markdown report
    markdown = f"# Prediction Analysis Report\n\n"
    markdown += f"## Summary\n\n"
    markdown += f"- Total samples: {summary['total_samples']}\n"
    markdown += f"- Average prediction length: {summary['avg_hyp_length']:.2f} words\n"
    markdown += f"- Average reference length: {summary['avg_label_length']:.2f} words\n"
    markdown += f"- Exact matches: {summary['exact_matches']} ({summary['exact_matches']/summary['total_samples']*100:.2f}%)\n\n"
    
    markdown += f"## Length Categories\n\n"
    for category, count in category_counts.items():
        markdown += f"- {category}: {count} ({count/summary['total_samples']*100:.2f}%)\n"
    
    markdown += f"\n## Examples\n\n"
    for category, samples in examples.items():
        markdown += f"### {category.title()} Predictions\n\n"
        for i, sample in enumerate(samples[:5]):  # Limit to 5 examples in the markdown
            markdown += f"**Example {i+1}**:\n\n"
            markdown += f"Prediction:\n```\n{sample['hyp']}\n```\n\n"
            markdown += f"Reference:\n```\n{sample['label']}\n```\n\n"
    
    # Save markdown report
    with open(os.path.join(output_dir, 'prediction_analysis.md'), 'w') as f:
        f.write(markdown)
    
    # Plot length difference histogram
    plt.figure(figsize=(10, 6))
    plt.hist(pred_df['length_diff'], bins=30)
    plt.title('Histogram of Length Difference (Prediction - Reference)')
    plt.xlabel('Length Difference (chars)')
    plt.ylabel('Count')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'length_diff_histogram.png'))
    plt.close()
    
    logger.info(f"Prediction analysis saved to {output_dir}/prediction_analysis.json and {output_dir}/prediction_analysis.md")

def push_model_to_hub(
    model_path: str, 
    tokenizer_path: str, 
    hub_model_id: str, 
    hub_token: Optional[str] = None, 
    metadata: Optional[Dict[str, Any]] = None
) -> None:
    """
    Push fine-tuned model to Hugging Face Hub.
    
    Args:
        model_path: Path to model directory or checkpoint
        tokenizer_path: Path to tokenizer directory 
        hub_model_id: Model ID for Hugging Face Hub
        hub_token: Hugging Face Hub token
        metadata: Additional model metadata
    """
    if hub_token:
        HfFolder.save_token(hub_token)
        
    # Load model and tokenizer
    logger.info(f"Loading model from {model_path}")
    model = MBartForConditionalGeneration.from_pretrained(model_path)
    
    # Set model card metadata
    if metadata is None:
        metadata = {}
    
    default_metadata = {
        "language": "id",
        "license": "apache-2.0",
        "tags": ["summarization", "indonesian", "indobart-v2", "indosum"],
        "datasets": ["indosum"],
        "metrics": ["bleu", "rouge"],
    }
    
    # Merge with user-provided metadata
    metadata = {**default_metadata, **metadata}
    
    # Push to Hub
    logger.info(f"Pushing model to Hub as {hub_model_id}")
    model.push_to_hub(
        hub_model_id,
        use_auth_token=hub_token or True,
        tags=metadata["tags"],
        commit_message="IndoBART-v2 fine-tuned on IndoSUM dataset"
    )
    
    # Create model card
    model_card = f"""
# IndoBART-v2 Fine-tuned on IndoSUM

This model is a fine-tuned version of [indobenchmark/indobart-v2](https://huggingface.co/indobenchmark/indobart-v2) on the IndoSUM dataset for Indonesian text summarization.

## Model description

IndoBART-v2 is a sequence-to-sequence model based on the BART architecture, pre-trained on a large corpus of Indonesian text. This version has been fine-tuned specifically for summarization tasks using the IndoSUM dataset.

## Intended uses & limitations

The model is intended for Indonesian text summarization. It can be used to generate concise summaries of longer Indonesian texts.

## Training procedure

The model was trained using the following parameters:
- Learning rate: 5e-5
- Batch size: 8
- Max sequence length: 512
- Early stopping based on SacreBLEU score

## Evaluation results

The model was evaluated on the IndoSUM test set:

{metadata.get('metrics_summary', 'Metrics information not available.')}

## Usage

```python
from transformers import MBartForConditionalGeneration
from indonlg.modules.tokenization_indonlg import IndoNLGTokenizer

# Load model and tokenizer
model = MBartForConditionalGeneration.from_pretrained("{hub_model_id}")
tokenizer = IndoNLGTokenizer.from_pretrained("{hub_model_id}")

# Prepare input
text = "Artikel ini membahas tentang perkembangan teknologi kecerdasan buatan di Indonesia selama dekade terakhir. Berbagai startup dan perusahaan teknologi telah mengembangkan solusi AI untuk berbagai masalah, mulai dari pertanian hingga layanan kesehatan. Pemerintah juga telah mendukung inisiatif ini melalui berbagai program pendanaan dan kebijakan yang mendukung inovasi di bidang kecerdasan buatan."
inputs = tokenizer("[indonesian] " + text, return_tensors="pt")

# Generate summary
output_ids = model.generate(
    inputs["input_ids"],
    max_length=150,
    num_beams=5,
    early_stopping=True,
    decoder_start_token_id=tokenizer.special_tokens_to_ids["[indonesian]"]
)

# Decode and print summary
summary = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(summary)
```

## Limitations and biases

The model may inherit biases from the pre-training data and the fine-tuning dataset. It may struggle with highly technical or domain-specific content that was underrepresented in the training data.
"""
    
    # Save model card
    with open("README.md", "w") as f:
        f.write(model_card)
    
    # Push readme
    api = HfApi()
    api.upload_file(
        path_or_fileobj="README.md",
        path_in_repo="README.md",
        repo_id=hub_model_id,
        repo_type="model",
        use_auth_token=hub_token or True,
    )
    
    logger.info(f"Model pushed to {hub_model_id}")

def load_checkpoint(
    checkpoint_path: str, 
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None
) -> Tuple[torch.nn.Module, Optional[torch.optim.Optimizer], Optional[Any], int, Dict[str, float]]:
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load checkpoint into
        optimizer: Optimizer to load checkpoint into
        scheduler: Scheduler to load checkpoint into
        
    Returns:
        Tuple of (model, optimizer, scheduler, epoch, metrics)
    """
    if not os.path.exists(checkpoint_path):
        logger.error(f"Checkpoint file {checkpoint_path} not found.")
        return model, optimizer, scheduler, 0, {}
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler is not None and 'scheduler_state_dict' in checkpoint and checkpoint['scheduler_state_dict'] is not None:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Get epoch and metrics
    epoch = checkpoint.get('epoch', 0)
    val_metrics = checkpoint.get('val_metrics', {})
    
    logger.info(f"Loaded checkpoint from epoch {epoch} with metrics: {val_metrics}")
    
    return model, optimizer, scheduler, epoch, val_metrics

def find_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint in the directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if not found
    """
    if not os.path.exists(checkpoint_dir):
        logger.error(f"Checkpoint directory {checkpoint_dir} not found.")
        return None
    
    # List all checkpoint files
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith('checkpoint-') and f.endswith('.pt')]
    
    if not checkpoint_files:
        logger.warning(f"No checkpoint files found in {checkpoint_dir}.")
        return None
    
    # Extract epoch numbers
    epoch_nums = [int(f.split('-')[1].split('.')[0]) for f in checkpoint_files]
    
    # Find the latest checkpoint
    latest_epoch = max(epoch_nums)
    latest_file = f"checkpoint-{latest_epoch}.pt"
    
    return os.path.join(checkpoint_dir, latest_file)

def format_metrics_for_logging(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary for logging.
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Formatted string
    """
    return ' | '.join(f"{key}: {value:.4f}" for key, value in metrics.items())
