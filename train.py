"""
Training functionality for fine-tuning IndoBART on the IndoSUM dataset.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import os
import torch
import numpy as np
import random
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import logging

# Import custom tokenizer - using relative import since it's within the project
from indonlg.modules.tokenization_indonlg import IndoNLGTokenizer

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_lr(optimizer: torch.optim.Optimizer) -> float:
    """
    Get the current learning rate from the optimizer.
    
    Args:
        optimizer: Optimizer to get learning rate from
        
    Returns:
        Current learning rate
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def metrics_to_string(metric_dict: Dict[str, float]) -> str:
    """
    Convert metrics dictionary to a readable string.
    
    Args:
        metric_dict: Dictionary of metrics
        
    Returns:
        String representation of metrics
    """
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('%s:%.2f' % (key, value))
    return ' '.join(string_list)


def train(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    forward_fn: Callable,
    metrics_fn: Callable,
    valid_criterion: str,
    tokenizer: IndoNLGTokenizer,
    n_epochs: int,
    evaluate_every: int = 1,
    early_stop: int = 3,
    step_size: int = 1,
    gamma: float = 0.5,
    max_norm: float = 10,
    grad_accum: int = 1,
    beam_size: int = 5,
    max_seq_len: int = 512,
    model_type: str = 'indo-bart',
    model_dir: str = "./save",
    exp_id: Optional[str] = None,
    fp16: bool = False,
    device: str = 'cpu',
    checkpoint_callback: Optional[Callable] = None,
) -> Dict[str, Any]:
    """
    Train the model.
    
    Args:
        model: Model to train
        train_loader: DataLoader for training data
        valid_loader: DataLoader for validation data
        optimizer: Optimizer to use for training
        forward_fn: Function to use for forward pass
        metrics_fn: Function to use for metrics calculation
        valid_criterion: Metric to use for validation
        tokenizer: Tokenizer to use for preprocessing
        n_epochs: Number of epochs to train for
        evaluate_every: Evaluate every n epochs
        early_stop: Stop after n epochs without improvement
        step_size: Step size for learning rate scheduler
        gamma: Gamma for learning rate scheduler
        max_norm: Max norm for gradient clipping
        grad_accum: Gradient accumulation steps
        beam_size: Beam size for generation
        max_seq_len: Maximum sequence length
        model_type: Type of model being used
        model_dir: Directory to save model to
        exp_id: Experiment ID
        fp16: Whether to use FP16 precision
        device: Device to use for training
        checkpoint_callback: Callback for checkpoint saving
        
    Returns:
        Dictionary of training history
    """
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    # Initialize variables
    best_val_metric = -100
    count_stop = 0
    history = {
        'train_loss': [],
        'train_metrics': [],
        'val_loss': [],
        'val_metrics': [],
        'learning_rates': []
    }
    
    if fp16:
        scaler = torch.cuda.amp.GradScaler()
    
    # Start training
    logger.info("Starting training...")
    
    for epoch in range(n_epochs):
        model.train()
        torch.set_grad_enabled(True)
        
        total_train_loss = 0
        list_hyp, list_label = [], []
        
        train_pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
        for i, batch_data in enumerate(train_pbar):
            if fp16:
                with torch.cuda.amp.autocast():
                    loss, batch_hyp, batch_label = forward_fn(
                        model, batch_data, model_type=model_type, tokenizer=tokenizer,
                        device=device, skip_special_tokens=False, is_test=False
                    )
                    
                # Scales the loss, and calls backward() to create scaled gradients
                scaler.scale(loss).backward()
                
                # Unscales the gradients of optimizer's assigned params in-place
                scaler.unscale_(optimizer)
                
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
                
                # Unscales gradients and calls or skips optimizer.step()
                scaler.step(optimizer)
                
                # Updates the scale for next iteration
                scaler.update()
            else:
                loss, batch_hyp, batch_label = forward_fn(
                    model, batch_data, model_type=model_type, tokenizer=tokenizer,
                    device=device, skip_special_tokens=False, is_test=False
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            
            tr_loss = loss.item()
            total_train_loss = total_train_loss + tr_loss
            
            # Calculate metrics
            list_hyp += batch_hyp
            list_label += batch_label
            
            current_lr = get_lr(optimizer)
            train_pbar.set_description(
                "(Epoch %d) TRAIN LOSS:%.4f LR:%.8f" % (epoch+1, total_train_loss/(i+1), current_lr)
            )
            
            if (i + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
        
        # Compute training metrics
        metrics = metrics_fn(list_hyp, list_label)
        
        logger.info(
            "(Epoch %d) TRAIN LOSS:%.4f %s LR:%.8f",
            epoch+1, total_train_loss/(i+1), metrics_to_string(metrics), current_lr
        )
        
        # Store training history
        history['train_loss'].append(total_train_loss / (i + 1))
        history['train_metrics'].append(metrics)
        history['learning_rates'].append(current_lr)
        
        # Decay Learning Rate
        scheduler.step()
        
        # Evaluate
        if ((epoch + 1) % evaluate_every) == 0:
            # Explicitly specifying that we only want the first two return values for validation
            # This makes the linter understand we're deliberately using only 2 of the possible 4 return values
            val_results = evaluate(
                model, valid_loader, forward_fn, metrics_fn, model_type,
                tokenizer, is_test=False, beam_size=beam_size,
                max_seq_len=max_seq_len, device=device
            )
            val_loss, val_metrics = val_results
            
            logger.info(
                "(Epoch %d) VALID LOSS:%.4f %s",
                epoch+1, val_loss, metrics_to_string(val_metrics)
            )
            
            # Store validation history
            history['val_loss'].append(val_loss)
            history['val_metrics'].append(val_metrics)
            
            # Early stopping
            val_metric = val_metrics[valid_criterion]
            if best_val_metric < val_metric:
                best_val_metric = val_metric
                # Save model
                if exp_id is not None:
                    checkpoint_path = os.path.join(model_dir, f"best_model_{exp_id}.pt")
                else:
                    checkpoint_path = os.path.join(model_dir, "best_model.pt")
                
                torch.save(model.state_dict(), checkpoint_path)
                count_stop = 0
                
                if checkpoint_callback:
                    checkpoint_callback(model, epoch, checkpoint_path, val_metric)
            else:
                count_stop += 1
                logger.info("count stop: %d", count_stop)
                if count_stop == early_stop:
                    logger.info("Early stopping triggered after %d epochs", epoch+1)
                    break
    
    return history


def evaluate(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    forward_fn: Callable,
    metrics_fn: Callable,
    model_type: str,
    tokenizer: IndoNLGTokenizer,
    beam_size: int = 5,
    max_seq_len: int = 512,
    is_test: bool = False,
    device: str = 'cpu',
    length_penalty: float = 1.0,
) -> Union[Tuple[float, Dict[str, float]], Tuple[float, Dict[str, float], List[str], List[str]]]:
    """
    Evaluate the model on the given data.
    
    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation
        forward_fn: Function to use for forward pass
        metrics_fn: Function to compute metrics
        model_type: Type of model
        tokenizer: Tokenizer for preprocessing
        beam_size: Beam size for generation
        max_seq_len: Maximum sequence length
        is_test: Whether this is a test evaluation
        device: Device to use
        length_penalty: Length penalty for generation
        
    Returns:
        If is_test=False: Tuple of (loss, metrics)
        If is_test=True: Tuple of (loss, metrics, hypotheses, references)
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
            
            pbar.set_description("VALID LOSS:%.4f" % (total_loss/(i+1)))
        else:
            pbar.set_description("TESTING... ")
    
    metrics = metrics_fn(list_hyp, list_label)
    
    # Store i value in case the loop is empty
    i_value = max(0, i) if 'i' in locals() else 0
    
    if is_test:
        # Return 4 values for test mode
        return total_loss/(i_value+1), metrics, list_hyp, list_label
    else:
        # Return only 2 values for validation mode
        return total_loss/(i_value+1), metrics
