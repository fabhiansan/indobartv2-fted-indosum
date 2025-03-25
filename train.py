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
from transformers import MBartForConditionalGeneration, get_linear_schedule_with_warmup
# from transformers.file_utils import is_torch_tpu_available
from huggingface_hub import HfApi

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import IndoNLG utilities
try:
    from indonlg.modules.tokenization_indonlg import IndoNLGTokenizer
    from indonlg.utils.metrics import generation_metrics_fn 
    from indonlg.utils.forward_fn import forward_generation
    from indonlg.utils.data_utils import MachineTranslationDataset, GenerationDataLoader
except ImportError:
    # Alternative path if above imports fail
    from indobenchmark import IndoNLGTokenizer
    from indonlg.utils.metrics import generation_metrics_fn
    from indonlg.utils.forward_fn import forward_generation
    from indonlg.utils.data_utils import MachineTranslationDataset, GenerationDataLoader

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
        loss, batch_hyp, batch_label = forward_fn(
            model, batch_data, model_type=model_type, tokenizer=tokenizer, 
            device=device, is_inference=is_test, is_test=is_test, 
            skip_special_tokens=True, beam_size=beam_size, 
            max_seq_len=max_seq_len, length_penalty=length_penalty
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
                        model, batch_data, model_type=model_type, tokenizer=tokenizer, 
                        device=device, skip_special_tokens=False, is_test=False
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
                    model, batch_data, model_type=model_type, tokenizer=tokenizer, 
                    device=device, skip_special_tokens=False, is_test=False
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
                model, valid_loader, forward_fn, metrics_fn, model_type, tokenizer, 
                is_test=False, beam_size=beam_size, max_seq_len=max_seq_len, device=device
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

def main():
    parser = argparse.ArgumentParser(description="Fine-tune IndoBART-v2 on IndoSUM")
    
    # Model and training arguments
    parser.add_argument("--model_name", type=str, default="indobenchmark/indobart-v2", 
                        help="Pretrained model name or path")
    parser.add_argument("--output_dir", type=str, default="./checkpoints/indosum", 
                        help="Directory to save model checkpoints")
    parser.add_argument("--dataset_dir", type=str, default="./dataset/IndoSUM", 
                        help="Directory containing IndoSUM dataset")
    parser.add_argument("--train_file", type=str, default="train_preprocess.json", 
                        help="Training file name")
    parser.add_argument("--valid_file", type=str, default="valid_preprocess.json", 
                        help="Validation file name")
    parser.add_argument("--test_file", type=str, default="test_preprocess.json", 
                        help="Test file name")
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
    
    # Language tags
    parser.add_argument("--source_lang", type=str, default="[indonesian]", 
                        help="Source language token")
    parser.add_argument("--target_lang", type=str, default="[indonesian]", 
                        help="Target language token")
    
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
    tokenizer = IndoNLGTokenizer.from_pretrained(args.model_name)
    logger.info(f"Model loaded with {count_param(model)} parameters")
    
    # Setup special tokens and language IDs
    src_lid = tokenizer.special_tokens_to_ids[args.source_lang]
    tgt_lid = tokenizer.special_tokens_to_ids[args.target_lang]
    
    # Inject language ID as BOS token in model.generate() function
    tokenizer.bos_token = args.target_lang
    model.config.decoder_start_token_id = tgt_lid
    
    # Check CUDA availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    model.to(device)
    
    # Create dataset and data loaders
    logger.info("Loading datasets")
    train_path = os.path.join(args.dataset_dir, args.train_file)
    valid_path = os.path.join(args.dataset_dir, args.valid_file)
    test_path = os.path.join(args.dataset_dir, args.test_file)
    
    # Check if dataset exists
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training file not found at {train_path}")
    if not os.path.exists(valid_path):
        raise FileNotFoundError(f"Validation file not found at {valid_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found at {test_path}")
    
    # Create datasets
    train_dataset = MachineTranslationDataset(
        train_path, tokenizer, lowercase=True, no_special_token=False,
        speaker_1_id=5, speaker_2_id=6, separator_id=4,
        max_token_length=args.max_seq_length, swap_source_target=False
    )
    valid_dataset = MachineTranslationDataset(
        valid_path, tokenizer, lowercase=True, no_special_token=False,
        speaker_1_id=5, speaker_2_id=6, separator_id=4,
        max_token_length=args.max_seq_length, swap_source_target=False
    )
    test_dataset = MachineTranslationDataset(
        test_path, tokenizer, lowercase=True, no_special_token=False,
        speaker_1_id=5, speaker_2_id=6, separator_id=4,
        max_token_length=args.max_seq_length, swap_source_target=False
    )
    
    # Create data loaders
    train_loader = GenerationDataLoader(
        dataset=train_dataset, model_type='indo-bart', tokenizer=tokenizer, 
        max_seq_len=args.max_seq_length, batch_size=args.train_batch_size,
        src_lid_token_id=src_lid, tgt_lid_token_id=tgt_lid, num_workers=4, shuffle=True
    )
    valid_loader = GenerationDataLoader(
        dataset=valid_dataset, model_type='indo-bart', tokenizer=tokenizer, 
        max_seq_len=args.max_seq_length, batch_size=args.valid_batch_size,
        src_lid_token_id=src_lid, tgt_lid_token_id=tgt_lid, num_workers=4, shuffle=False
    )
    test_loader = GenerationDataLoader(
        dataset=test_dataset, model_type='indo-bart', tokenizer=tokenizer, 
        max_seq_len=args.max_seq_length, batch_size=args.test_batch_size,
        src_lid_token_id=src_lid, tgt_lid_token_id=tgt_lid, num_workers=4, shuffle=False
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
        forward_fn=forward_generation,
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
        model, test_loader, forward_generation, generation_metrics_fn, 
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
