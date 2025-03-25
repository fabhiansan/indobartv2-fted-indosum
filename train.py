#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Fine-tuning script for IndoBART-v2 on IndoSUM dataset.
This script trains the model, evaluates after each checkpoint,
and pushes the final model to Hugging Face Hub.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
from transformers import MBartForConditionalGeneration, get_linear_schedule_with_warmup, MBartTokenizer

# Handle different transformers versions for TPU availability check
try:
    from transformers.utils.imports import is_torch_tpu_available
except ImportError:
    try:
        from transformers.file_utils import is_torch_tpu_available
    except ImportError:
        # Fallback if neither import works
        def is_torch_tpu_available():
            return False

from huggingface_hub import HfApi
from datasets import load_from_disk, concatenate_datasets
from torch.utils.data import DataLoader, Dataset

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import IndoNLG utilities
try:
    from transformers import MBartTokenizer, MBartForConditionalGeneration
    from indonlg.utils.metrics import generation_metrics_fn 
    from indonlg.utils.forward_fn import forward_generation
except ImportError:
    # Alternative path if above imports fail
    try:
        from transformers import MBartTokenizer, MBartForConditionalGeneration
        from indonlg.utils.metrics import generation_metrics_fn
        from indonlg.utils.forward_fn import forward_generation
    except ImportError:
        raise ImportError(
            "Could not import required modules. Make sure transformers and indonlg are installed."
        )

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make CuDNN behavior deterministic
    torch.backends.cudnn.deterministic = True

def count_param(module: torch.nn.Module, trainable: bool = False) -> int:
    """
    Count parameters in a model.
    
    Args:
        module: PyTorch model
        trainable: If True, count only trainable parameters
        
    Returns:
        Number of parameters
    """
    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())

def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get the current learning rate from optimizer.
    
    Args:
        optimizer: PyTorch optimizer
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def metrics_to_string(metric_dict: Dict[str, float]) -> str:
    """
    Convert metrics dictionary to a formatted string.
    
    Args:
        metric_dict: Dictionary of metrics
        
    Returns:
        Formatted string of metrics
    """
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)

def evaluate(
    model: torch.nn.Module, 
    data_loader: torch.utils.data.DataLoader, 
    forward_fn: callable, 
    metrics_fn: callable, 
    model_type: str, 
    tokenizer: Any, 
    beam_size: int = 5, 
    max_seq_len: int = 512, 
    is_test: bool = False, 
    device: str = 'cpu',
    length_penalty: float = 1.0
) -> Union[Tuple[float, Dict[str, float]], Tuple[float, Dict[str, float], List[str], List[str]]]:
    """
    Evaluate model on given dataset.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation
        forward_fn: Function to perform forward pass
        metrics_fn: Function to calculate metrics
        model_type: Type of model ('indo-bart', etc.)
        tokenizer: Tokenizer for decoding
        beam_size: Beam size for generation
        max_seq_len: Maximum sequence length
        is_test: Whether this is test evaluation
        device: Device to use
        length_penalty: Length penalty for generation
        
    Returns:
        If is_test=False: Tuple of (loss, metrics)
        If is_test=True: Tuple of (loss, metrics, hypotheses, labels)
    """
    model.eval()
    torch.set_grad_enabled(False)
    
    total_loss = 0
    list_hyp, list_label = [], []

    pbar = tqdm(iter(data_loader), leave=True, total=len(data_loader))
    for i, batch_data in enumerate(pbar):
        # Call the forward function
        loss, batch_hyp, batch_label = forward_fn(
            model=model, 
            batch=batch_data, 
            device=device, 
            is_inference=True, 
            is_test=is_test,
            tokenizer=tokenizer,
            beam_size=beam_size,
            max_gen_length=max_seq_len,
            min_gen_length=10
        )
        
        # Calculate evaluation metrics
        list_hyp += batch_hyp
        list_label += batch_label

        if not is_test:
            # Calculate total loss for validation
            test_loss = loss.item()
            total_loss = total_loss + test_loss
            pbar.set_description(f"VALID LOSS:{total_loss/(i+1):.4f}")
        else:
            pbar.set_description("TESTING... ")
    
    metrics = metrics_fn(list_hyp, list_label)
    if is_test:
        return total_loss/(i+1) if i > 0 else total_loss, metrics, list_hyp, list_label
    else:
        return total_loss/(i+1) if i > 0 else total_loss, metrics

def train(
    model: torch.nn.Module, 
    train_loader: torch.utils.data.DataLoader, 
    valid_loader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    forward_fn: callable, 
    metrics_fn: callable, 
    valid_criterion: str, 
    tokenizer: Any, 
    n_epochs: int, 
    evaluate_every: int = 1, 
    early_stop: int = 3, 
    max_norm: float = 10, 
    grad_accum: int = 1, 
    beam_size: int = 5, 
    max_seq_len: int = 512, 
    model_type: str = 'indo-bart', 
    model_dir: str = "", 
    exp_id: Optional[str] = None, 
    fp16: bool = False, 
    device: str = 'cpu',
    push_to_hub: bool = False,
    hub_model_id: Optional[str] = None,
    hub_token: Optional[str] = None,
    evaluate_during_training: bool = True
) -> List[Dict[str, float]]:
    """
    Train model and evaluate on validation set.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training
        valid_loader: DataLoader for validation
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        forward_fn: Function to perform forward pass
        metrics_fn: Function to calculate metrics
        valid_criterion: Criterion to use for validation (e.g. 'SacreBLEU')
        tokenizer: Tokenizer
        n_epochs: Number of epochs
        evaluate_every: Evaluate every n epochs
        early_stop: Early stopping patience
        max_norm: Max norm for gradient clipping
        grad_accum: Gradient accumulation steps
        beam_size: Beam size for evaluation
        max_seq_len: Maximum sequence length
        model_type: Type of model ('indo-bart', etc.)
        model_dir: Directory to save model
        exp_id: Experiment ID
        fp16: Whether to use mixed precision
        device: Device to use
        push_to_hub: Whether to push to Hub after training
        hub_model_id: Model ID for Hub
        hub_token: Hub API token
        evaluate_during_training: Whether to evaluate during training
        
    Returns:
        List of validation metrics for each evaluated epoch
    """
    if fp16:
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
        except ImportError:
            raise ImportError("Mixed precision training requires PyTorch >= 1.6")
    
    best_val_metric = -100
    count_stop = 0
    all_val_metrics = []
    
    for epoch in range(n_epochs):
        model.train()
        torch.set_grad_enabled(True)
        
        total_train_loss = 0
        list_hyp, list_label = [], []
        
        train_pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            if fp16:
                with autocast():
                    loss, batch_hyp, batch_label = forward_fn(
                        model=model,
                        batch=batch_data,
                        device=device,
                        is_inference=False,
                        is_test=False,
                        tokenizer=tokenizer
                    )
                
                # Scale the loss and call backward()
                scaler.scale(loss).backward()
                
                if (i + 1) % grad_accum == 0:
                    # Unscale the gradients
                    scaler.unscale_(optimizer)
                    
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    
                    # Update parameters
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss, batch_hyp, batch_label = forward_fn(
                    model=model,
                    batch=batch_data,
                    device=device,
                    is_inference=False,
                    is_test=False,
                    tokenizer=tokenizer
                )
                
                loss = loss / grad_accum
                loss.backward()
                
                if (i + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                    optimizer.step()
                    optimizer.zero_grad()
            
            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss
            
            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label
            
            train_pbar.set_description(
                f"(Epoch {epoch+1}) TRAIN LOSS:{total_train_loss/(i+1):.4f} LR:{get_lr(optimizer):.8f}"
            )
        
        # Calculate training metrics
        train_metrics = metrics_fn(list_hyp, list_label)
        logger.info(
            f"(Epoch {epoch+1}) TRAIN LOSS:{total_train_loss/(i+1):.4f} "
            f"{metrics_to_string(train_metrics)} LR:{get_lr(optimizer):.8f}"
        )
        
        # Update learning rate
        scheduler.step()
        
        # Evaluate
        if evaluate_during_training and ((epoch+1) % evaluate_every) == 0:
            val_loss, val_metrics = evaluate(
                model=model, 
                data_loader=valid_loader, 
                forward_fn=forward_fn, 
                metrics_fn=metrics_fn, 
                model_type=model_type, 
                tokenizer=tokenizer, 
                is_test=False, 
                beam_size=beam_size, 
                max_seq_len=max_seq_len, 
                device=device
            )
            
            all_val_metrics.append(val_metrics)
            
            logger.info(
                f"(Epoch {epoch+1}) VALID LOSS:{val_loss:.4f} {metrics_to_string(val_metrics)}"
            )
            
            # Save checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'val_metrics': val_metrics,
                'val_loss': val_loss
            }
            
            checkpoint_path = os.path.join(model_dir, f"checkpoint-{epoch+1}.pt")
            torch.save(checkpoint, checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")
            
            # Early stopping
            val_metric = val_metrics[valid_criterion]
            if best_val_metric < val_metric:
                best_val_metric = val_metric
                # Save best model
                if exp_id is not None:
                    torch.save(model.state_dict(), os.path.join(model_dir, f"best_model_{exp_id}.pt"))
                else:
                    torch.save(model.state_dict(), os.path.join(model_dir, "best_model.pt"))
                count_stop = 0
            else:
                count_stop += 1
                logger.info(f"Early stopping counter: {count_stop}/{early_stop}")
                if count_stop == early_stop:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
    
    # Push to Hub if requested
    if push_to_hub and hub_model_id and hub_token:
        logger.info(f"Pushing model to Hub as {hub_model_id}")
        model.push_to_hub(
            hub_model_id,
            use_auth_token=hub_token,
            tags=["summarization", "indonesian", "indobart-v2", "indosum"],
            commit_message="IndoBART-v2 fine-tuned on IndoSUM dataset"
        )
        tokenizer.push_to_hub(
            hub_model_id,
            use_auth_token=hub_token
        )
    
    return all_val_metrics

class IndoSUMDataset(Dataset):
    """
    Custom dataset for IndoSUM dataset processing.
    
    This dataset handles the tokenization of input documents and target summaries.
    """
    
    def __init__(
        self, 
        dataset: Union['Dataset', List[Dict[str, str]]],
        tokenizer: 'MBartTokenizer',
        source_column: str = 'article',
        target_column: str = 'summary',
        source_lang: str = '[indonesian]',
        target_lang: str = '[indonesian]',
        max_source_length: int = 1024,
        max_target_length: int = 128
    ):
        """
        Initialize IndoSUMDataset.
        
        Args:
            dataset: HuggingFace dataset or list of dictionaries
            tokenizer: Tokenizer for encoding inputs
            source_column: Column name for source text
            target_column: Column name for target text
            source_lang: Source language token
            target_lang: Target language token
            max_source_length: Maximum length for source text
            max_target_length: Maximum length for target text
        """
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.source_column = source_column
        self.target_column = target_column
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        
        # Set src and tgt language for MBartTokenizer
        if hasattr(tokenizer, 'lang_code_to_id'):
            # Use standard MBart language codes
            self.mbart_mode = True
            if isinstance(source_lang, str) and source_lang.startswith('['):
                # Convert indonesian token to MBart language code
                self.src_lang_code = "id_ID"
                self.tgt_lang_code = "id_ID"
            else:
                self.src_lang_code = source_lang
                self.tgt_lang_code = target_lang
                
            # Set the current language for the tokenizer
            self.tokenizer.src_lang = self.src_lang_code
            self.tokenizer.tgt_lang = self.tgt_lang_code
            
            # Get language IDs
            self.src_lid = tokenizer.lang_code_to_id[self.src_lang_code]
            self.tgt_lid = tokenizer.lang_code_to_id[self.tgt_lang_code]
        else:
            # Use special tokens mode for custom tokenizers
            self.mbart_mode = False
            if hasattr(tokenizer, 'special_tokens_to_ids'):
                self.src_lid = tokenizer.special_tokens_to_ids.get(source_lang, tokenizer.pad_token_id)
                self.tgt_lid = tokenizer.special_tokens_to_ids.get(target_lang, tokenizer.pad_token_id)
            else:
                # Fallback to pad token ID
                self.src_lid = tokenizer.pad_token_id
                self.tgt_lid = tokenizer.pad_token_id
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example from the dataset.
        
        Args:
            idx: Index of the example
            
        Returns:
            Dictionary with preprocessed source and target texts
        """
        item = self.dataset[idx]
        source_text = item[self.source_column]
        target_text = item[self.target_column]
        
        # Tokenize source text
        if self.mbart_mode:
            # For standard MBart tokenizer with language codes
            src_inputs = self.tokenizer(
                source_text,
                max_length=self.max_source_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                src_lang=self.src_lang_code
            )
            
            # Tokenize target text without the context manager
            tgt_inputs = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt',
                tgt_lang=self.tgt_lang_code,
                add_special_tokens=True,
                # Ensure we're in target mode
                forced_bos_token_id=self.tgt_lid
            )
        else:
            # For custom tokenizers
            # Prepend language tokens if using special tokens
            if self.source_lang.startswith('['):
                source_text = f"{self.source_lang} {source_text}"
                target_text = f"{self.target_lang} {target_text}"
                
            src_inputs = self.tokenizer(
                source_text,
                max_length=self.max_source_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # For non-MBart tokenizers, we don't need the context manager
            tgt_inputs = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        
        # Extract tensors and convert to appropriate shape
        source_ids = src_inputs['input_ids'].squeeze()
        source_mask = src_inputs['attention_mask'].squeeze()
        target_ids = tgt_inputs['input_ids'].squeeze()
        
        # Replace padding with -100 for loss calculation
        labels = target_ids.clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'source_ids': source_ids,
            'source_mask': source_mask,
            'target_ids': target_ids,
            'labels': labels,
            'source_text': source_text,
            'target_text': target_text
        }

def forward_pass(
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    device: str,
    is_inference: bool = False,
    is_test: bool = False,
    tokenizer: Optional[Any] = None,
    beam_size: int = 5,
    max_gen_length: int = 128,
    min_gen_length: int = 10
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """
    Perform forward pass for the model.
    
    Args:
        model: The model to perform forward pass
        batch: Input batch containing tokenized inputs
        device: Device to run on
        is_inference: Whether to run in inference mode
        is_test: Whether this is for testing
        tokenizer: Tokenizer for decoding outputs
        beam_size: Beam size for beam search
        max_gen_length: Maximum generation length
        min_gen_length: Minimum generation length
        
    Returns:
        Tuple containing loss, hypotheses, and labels
    """
    # Move batch data to device
    source_ids = batch["source_ids"].to(device)
    source_mask = batch["source_mask"].to(device)
    
    # Get source and target texts
    source_texts = batch["source_text"]
    target_texts = batch["target_text"]
    
    if is_inference:
        # Generate translations with beam search
        generated_ids = model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=beam_size,
            max_length=max_gen_length,
            min_length=min_gen_length,
            early_stopping=True,
            no_repeat_ngram_size=3,
            length_penalty=1.0
        )
        
        # Decode generated ids
        hypotheses = []
        for g in generated_ids:
            # Skip special tokens when decoding
            hyp = tokenizer.decode(g, skip_special_tokens=True)
            hypotheses.append(hyp)
        
        # Return placeholder loss (not used in inference)
        return torch.tensor(0.0), hypotheses, target_texts
    else:
        # Get labels for loss calculation
        labels = batch["labels"].to(device)
        
        # Forward pass to get loss
        outputs = model(
            input_ids=source_ids,
            attention_mask=source_mask,
            labels=labels,
            return_dict=True
        )
        
        loss = outputs.loss
        
        if is_test:
            # For testing, generate with beam search even during training forward pass
            generated_ids = model.generate(
                input_ids=source_ids,
                attention_mask=source_mask,
                num_beams=beam_size,
                max_length=max_gen_length,
                min_length=min_gen_length,
                early_stopping=True,
                no_repeat_ngram_size=3,
                length_penalty=1.0
            )
            
            # Decode generated ids
            hypotheses = []
            for g in generated_ids:
                hyp = tokenizer.decode(g, skip_special_tokens=True)
                hypotheses.append(hyp)
        else:
            # During training, use target text directly
            hypotheses = target_texts
            
        return loss, hypotheses, target_texts

def main():
    parser = argparse.ArgumentParser(description="Fine-tune IndoBART-v2 on IndoSUM")
    
    # Model and training arguments
    parser.add_argument("--model_name", type=str, default="indobenchmark/indobart-v2", 
                        help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/indosum", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--dataset_dir", type=str, default="../dataset/indosum", 
                        help="Directory containing IndoSUM dataset")
    parser.add_argument("--train_dataset_dir", type=str, default="traindataset", 
                        help="Directory containing training dataset")
    parser.add_argument("--valid_dataset_dir", type=str, default="devdataset", 
                        help="Directory containing validation dataset")
    parser.add_argument("--test_dataset_dir", type=str, default="testdataset", 
                        help="Directory containing test dataset")
    parser.add_argument("--source_column", type=str, default="document", 
                        help="Column name for source documents")
    parser.add_argument("--target_column", type=str, default="summary", 
                        help="Column name for target summaries")
    parser.add_argument("--source_lang", type=str, default="id_ID", 
                        help="Source language code")
    parser.add_argument("--target_lang", type=str, default="id_ID", 
                        help="Target language code")
    parser.add_argument("--num_epochs", type=int, default=10, 
                        help="Number of training epochs")
    parser.add_argument("--train_batch_size", type=int, default=8, 
                        help="Training batch size")
    parser.add_argument("--valid_batch_size", type=int, default=8, 
                        help="Validation batch size")
    parser.add_argument("--test_batch_size", type=int, default=8, 
                        help="Test batch size")
    parser.add_argument("--learning_rate", type=float, default=5e-5, 
                        help="Initial learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, 
                        help="Weight decay")
    parser.add_argument("--warmup_steps", type=int, default=0, 
                        help="Linear warmup steps")
    parser.add_argument("--gamma", type=float, default=0.9, 
                        help="Gamma for learning rate scheduler")
    parser.add_argument("--step_size", type=int, default=1, 
                        help="Step size for learning rate scheduler")
    parser.add_argument("--max_seq_length", type=int, default=512, 
                        help="Maximum sequence length")
    parser.add_argument("--beam_size", type=int, default=5, 
                        help="Beam size for generation")
    parser.add_argument("--grad_accum_steps", type=int, default=1, 
                        help="Gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=10.0, 
                        help="Maximum gradient norm")
    parser.add_argument("--early_stop", type=int, default=5, 
                        help="Early stopping patience")
    parser.add_argument("--valid_criterion", type=str, default="SacreBLEU", 
                        help="Validation criterion")
    parser.add_argument("--fp16", action="store_true", 
                        help="Use mixed precision training")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed")
    parser.add_argument("--force", action="store_true", 
                        help="Overwrite output directory if exists")
    
    # Hugging Face Hub arguments
    parser.add_argument("--push_to_hub", action="store_true", 
                        help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None, 
                        help="Model ID for Hugging Face Hub")
    parser.add_argument("--hub_token", type=str, default=None, 
                        help="Hugging Face Hub token")
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=args.force)
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.force:
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --force to overwrite."
        )
    
    # Load model and tokenizer
    logger.info(f"Loading model and tokenizer from {args.model_name}")
    model = MBartForConditionalGeneration.from_pretrained(args.model_name)
    
    # Use MBartTokenizer instead of IndoNLGTokenizer
    try:
        logger.info(f"Initializing tokenizer with src_lang={args.source_lang}, tgt_lang={args.target_lang}")
        # Try initializing with explicit src_lang and tgt_lang
        tokenizer = MBartTokenizer.from_pretrained(
            args.model_name, 
            src_lang=args.source_lang,
            tgt_lang=args.target_lang
        )
        
        # For compatibility, make sure both languages are set
        if not hasattr(tokenizer, 'src_lang') or tokenizer.src_lang is None:
            tokenizer.src_lang = args.source_lang
        if not hasattr(tokenizer, 'tgt_lang') or tokenizer.tgt_lang is None:
            tokenizer.tgt_lang = args.target_lang
            
        # Log the tokenizer configuration
        logger.info(f"Successfully initialized tokenizer: src_lang={tokenizer.src_lang}, tgt_lang={tokenizer.tgt_lang}")
        
        # Create dictionary for special tokens if needed
        if not hasattr(tokenizer, 'special_tokens_to_ids'):
            tokenizer.special_tokens_to_ids = {}
            
        # Set up special token mapping for Indonesian
        tokenizer.special_tokens_to_ids[args.source_lang] = tokenizer.lang_code_to_id[args.source_lang]
        tokenizer.special_tokens_to_ids[args.target_lang] = tokenizer.lang_code_to_id[args.target_lang]
        
    except Exception as e:
        logger.error(f"Error initializing tokenizer with language codes: {str(e)}")
        logger.info("Attempting to initialize without language codes...")
        
        # Try initializing without language codes
        tokenizer = MBartTokenizer.from_pretrained(args.model_name)
        
        # Set language codes after initialization
        try:
            tokenizer.src_lang = args.source_lang
            tokenizer.tgt_lang = args.target_lang
            
            # Set up special tokens for Indonesian
            special_tokens = {'additional_special_tokens': ['[indonesian]']}
            tokenizer.add_special_tokens(special_tokens)
            model.resize_token_embeddings(len(tokenizer))
            
            # Create mapping for special tokens
            tokenizer.special_tokens_to_ids = {
                args.source_lang: tokenizer.convert_tokens_to_ids('[indonesian]'),
                args.target_lang: tokenizer.convert_tokens_to_ids('[indonesian]')
            }
        except Exception as e2:
            logger.error(f"Error setting language attributes: {str(e2)}")
            raise RuntimeError("Could not initialize tokenizer properly. Please check model compatibility.")
    
    logger.info(f"Model loaded with {count_param(model)} parameters")
    
    # Verify that language IDs are properly set
    try:
        src_lid = tokenizer.special_tokens_to_ids[args.source_lang]
        tgt_lid = tokenizer.special_tokens_to_ids[args.target_lang]
        logger.info(f"Language ID tokens: src_lid={src_lid}, tgt_lid={tgt_lid}")
    except Exception as e:
        logger.error(f"Error accessing language IDs: {str(e)}")
        logger.info("Setting up default language IDs...")
        
        # Set up default language IDs
        src_lid = tokenizer.lang_code_to_id[args.source_lang]
        tgt_lid = tokenizer.lang_code_to_id[args.target_lang]
        tokenizer.special_tokens_to_ids = {
            args.source_lang: src_lid,
            args.target_lang: tgt_lid
        }
        logger.info(f"Set default language IDs: src_lid={src_lid}, tgt_lid={tgt_lid}")
    
    # Set decoder start token ID to target language ID
    model.config.decoder_start_token_id = tgt_lid
    logger.info(f"Set decoder_start_token_id to {tgt_lid}")
    
    # If using forced BOS token, verify it's set properly
    if hasattr(model.config, 'forced_bos_token_id'):
        model.config.forced_bos_token_id = tgt_lid
        logger.info(f"Set forced_bos_token_id to {tgt_lid}")
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Load datasets
    logger.info("Loading datasets")
    train_dataset_path = os.path.join(args.dataset_dir, args.train_dataset_dir)
    valid_dataset_path = os.path.join(args.dataset_dir, args.valid_dataset_dir)
    test_dataset_path = os.path.join(args.dataset_dir, args.test_dataset_dir)
    
    # Check if dataset directories exist
    if not os.path.exists(train_dataset_path):
        raise FileNotFoundError(f"Training dataset directory not found at {train_dataset_path}")
    if not os.path.exists(valid_dataset_path):
        raise FileNotFoundError(f"Validation dataset directory not found at {valid_dataset_path}")
    if not os.path.exists(test_dataset_path):
        raise FileNotFoundError(f"Test dataset directory not found at {test_dataset_path}")
    
    # Load datasets with error handling
    try:
        logger.info(f"Loading training dataset from {train_dataset_path}")
        train_hf_dataset = load_from_disk(train_dataset_path)
        logger.info(f"Loading validation dataset from {valid_dataset_path}")
        valid_hf_dataset = load_from_disk(valid_dataset_path)
        logger.info(f"Loading test dataset from {test_dataset_path}")
        test_hf_dataset = load_from_disk(test_dataset_path)
    except Exception as e:
        logger.error(f"Error loading datasets: {str(e)}")
        logger.info("Checking dataset structure...")
        # List contents of dataset directories to help diagnose
        for dataset_dir in [train_dataset_path, valid_dataset_path, test_dataset_path]:
            if os.path.exists(dataset_dir):
                logger.info(f"Contents of {dataset_dir}:")
                for item in os.listdir(dataset_dir):
                    logger.info(f"  - {item}")
        raise
    
    # Check if the expected columns exist
    expected_columns = [args.source_column, args.target_column]
    dataset_features = train_hf_dataset.features
    logger.info(f"Dataset features: {dataset_features}")
    for col in expected_columns:
        if col not in train_hf_dataset.column_names:
            available_cols = ', '.join(train_hf_dataset.column_names)
            raise ValueError(f"Column '{col}' not found in dataset. Available columns: {available_cols}")
    
    # Print dataset information
    logger.info(f"Train dataset size: {len(train_hf_dataset)}")
    logger.info(f"Validation dataset size: {len(valid_hf_dataset)}")
    logger.info(f"Test dataset size: {len(test_hf_dataset)}")
    
    # Show a sample from the dataset
    logger.info("Sample from training dataset:")
    sample = train_hf_dataset[0]
    for k, v in sample.items():
        logger.info(f"  {k}: {v[:100]}..." if isinstance(v, str) and len(v) > 100 else f"  {k}: {v}")
    
    # Create custom datasets
    train_dataset = IndoSUMDataset(
        train_hf_dataset, 
        tokenizer, 
        source_column=args.source_column, 
        target_column=args.target_column,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_source_length=args.max_seq_length,
        max_target_length=args.max_seq_length // 2
    )
    valid_dataset = IndoSUMDataset(
        valid_hf_dataset, 
        tokenizer, 
        source_column=args.source_column, 
        target_column=args.target_column,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_source_length=args.max_seq_length,
        max_target_length=args.max_seq_length // 2
    )
    test_dataset = IndoSUMDataset(
        test_hf_dataset, 
        tokenizer, 
        source_column=args.source_column, 
        target_column=args.target_column,
        source_lang=args.source_lang,
        target_lang=args.target_lang,
        max_source_length=args.max_seq_length,
        max_target_length=args.max_seq_length // 2
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.train_batch_size, 
        shuffle=True,
        num_workers=4
    )
    valid_loader = DataLoader(
        valid_dataset, 
        batch_size=args.valid_batch_size, 
        shuffle=False,
        num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=args.test_batch_size, 
        shuffle=False,
        num_workers=4
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Create scheduler
    if args.warmup_steps > 0:
        # Linear warmup with decay scheduler
        total_steps = len(train_loader) * args.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
        )
    else:
        # Step scheduler
        scheduler = StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    
    # Train the model
    logger.info("Starting training")
    val_metrics = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        forward_fn=forward_pass,
        metrics_fn=generation_metrics_fn,
        valid_criterion=args.valid_criterion,
        tokenizer=tokenizer,
        n_epochs=args.num_epochs,
        evaluate_every=1,
        early_stop=args.early_stop,
        max_norm=args.max_grad_norm,
        grad_accum=args.grad_accum_steps,
        beam_size=args.beam_size,
        max_seq_len=args.max_seq_length,
        model_type='indo-bart',
        model_dir=args.output_dir,
        exp_id=None,
        fp16=args.fp16,
        device=device,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        hub_token=args.hub_token
    )
    
    # Save validation metrics
    metrics_df = pd.DataFrame.from_records(val_metrics)
    metrics_df.to_csv(os.path.join(args.output_dir, "validation_metrics.csv"), index=False)
    
    # Load best model for final evaluation
    logger.info("Loading best model for final evaluation")
    best_model_path = os.path.join(args.output_dir, "best_model.pt")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))
    
    # Evaluate on test set
    logger.info("Evaluating on test set")
    test_loss, test_metrics, test_hyp, test_label = evaluate(
        model, test_loader, forward_pass, generation_metrics_fn, 
        'indo-bart', tokenizer, beam_size=args.beam_size, 
        max_seq_len=args.max_seq_length, is_test=True, device=device
    )
    
    logger.info(f"TEST METRICS: {metrics_to_string(test_metrics)}")
    
    # Save test results
    result_df = pd.DataFrame({
        'hyp': test_hyp,
        'label': test_label
    })
    result_df.to_csv(os.path.join(args.output_dir, "test_predictions.csv"), index=False)
    
    # Save test metrics
    test_metrics_df = pd.DataFrame([test_metrics])
    test_metrics_df.to_csv(os.path.join(args.output_dir, "test_metrics.csv"), index=False)
    
    logger.info("Training and evaluation completed")

if __name__ == "__main__":
    main()
