# Fine-tuning IndoBART-v2 on IndoSUM Dataset

## Project Overview
This project aims to fine-tune the IndoBART-v2 model on the IndoSUM dataset for Indonesian text summarization. The trained model will be evaluated after each checkpoint and finally pushed to the Hugging Face Hub.

## Implementation Plan

### 1. Environment Setup
- Import necessary libraries from PyTorch, Transformers, and IndoNLG
- Set up logging and seed for reproducibility

### 2. Model and Tokenizer Configuration
- Load pretrained IndoBART-v2 model and tokenizer
- Configure model parameters and optimizer

### 3. Dataset Preparation
- Set up the IndoSUM dataset (already downloaded)
- Create train/validation/test data loaders

### 4. Training Pipeline
- Implement the training loop with gradient accumulation
- Add evaluation metrics after each epoch (BLEU, SacreBLEU, ROUGE)
- Save checkpoints during training
- Add early stopping mechanism

### 5. Checkpoint Evaluation
- Evaluate model performance at each checkpoint
- Generate and save metrics
- Create prediction results and confusion matrices

### 6. Hugging Face Hub Integration
- Add model saving functionality
- Configure the Hugging Face Hub API
- Push the fine-tuned model to the Hub with appropriate metadata

## Code Structure
- `train.py`: Main script for training the model
- `utils.py`: Helper functions for metrics, evaluation, data loading
- `hf_utils.py`: Utilities for Hugging Face Hub integration

## Resource Files and Paths
- IndoBART-v2 base model: `indobenchmark/indobart-v2`
- IndoSUM dataset: Expected at `./dataset/IndoSUM/`
- Checkpoints: Will be saved at `./checkpoints/indosum/`

## Debugging Tips

### Common Issues and Solutions

#### Import and Path Issues
- Check `sys.path.append('../')` to ensure modules are found
- Verify IndoNLG tokenizer import paths

#### Model Loading Issues
- If tokenizer has issues, check for updates in the IndoNLG library
- If model loading fails, check GPU memory and reduce batch size

#### Training Problems
- Monitor loss values to detect NaN or extremely high values
- If training is unstable, reduce learning rate or add gradient clipping

#### Data Loading Issues
- Validate dataset format and paths
- Check max sequence length for long documents

#### CUDA Out of Memory
- Reduce batch size
- Implement gradient accumulation
- Use mixed precision training (fp16)

#### Evaluation Metrics
- Verify BLEU and ROUGE score calculations
- Check token normalization for Indonesian text

### Key Code Locations
- Training loop: Found in the `train.py` script
- Forward pass: Check `forward_generation` in utils
- Evaluation function: Review the `evaluate` function
- Metric calculation: See `generation_metrics_fn`
