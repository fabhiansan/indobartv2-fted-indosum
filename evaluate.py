"""
Evaluation functionality for IndoBART fine-tuned on the IndoSUM dataset.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, Callable

import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from transformers import MBartForConditionalGeneration

# Import custom tokenizer - using relative import since it's within the project
from indonlg.modules.tokenization_indonlg import IndoNLGTokenizer

# Import metrics calculation utilities
from rouge_score import rouge_scorer
import sacrebleu
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

logger = logging.getLogger(__name__)


def compute_bleu_score(hypotheses: List[str], references: List[str]) -> float:
    """
    Compute BLEU score for a list of hypotheses and references.
    
    Args:
        hypotheses: List of generated summaries
        references: List of reference summaries
        
    Returns:
        BLEU score
    """
    if len(hypotheses) != len(references):
        raise ValueError("Number of hypotheses and references must match")
    
    smooth = SmoothingFunction().method1
    total_score = 0
    
    for hyp, ref in zip(hypotheses, references):
        # Tokenize
        hyp_tokens = hyp.split()
        ref_tokens = [ref.split()]
        
        # Calculate BLEU score
        score = sentence_bleu(ref_tokens, hyp_tokens, smoothing_function=smooth)
        total_score += score
    
    return (total_score / len(hypotheses)) * 100


def compute_sacrebleu_score(hypotheses: List[str], references: List[str]) -> float:
    """
    Compute SacreBLEU score for a list of hypotheses and references.
    
    Args:
        hypotheses: List of generated summaries
        references: List of reference summaries
        
    Returns:
        SacreBLEU score
    """
    if len(hypotheses) != len(references):
        raise ValueError("Number of hypotheses and references must match")
    
    # Format references for sacrebleu
    refs = [references]  # sacrebleu expects a list of lists
    
    # Compute score
    bleu = sacrebleu.corpus_bleu(hypotheses, refs)
    
    return bleu.score


def compute_rouge_scores(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores for a list of hypotheses and references.
    
    Args:
        hypotheses: List of generated summaries
        references: List of reference summaries
        
    Returns:
        Dictionary of ROUGE scores
    """
    if len(hypotheses) != len(references):
        raise ValueError("Number of hypotheses and references must match")
    
    # Initialize scorer
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
    
    # Compute scores
    scores = {
        'ROUGE1': 0,
        'ROUGE2': 0,
        'ROUGEL': 0,
        'ROUGELsum': 0
    }
    
    for hyp, ref in zip(hypotheses, references):
        if not hyp.strip() or not ref.strip():
            continue
            
        score = scorer.score(ref, hyp)
        scores['ROUGE1'] += score['rouge1'].fmeasure
        scores['ROUGE2'] += score['rouge2'].fmeasure
        scores['ROUGEL'] += score['rougeL'].fmeasure
        scores['ROUGELsum'] += score['rougeLsum'].fmeasure
    
    # Average the scores
    n = len(hypotheses)
    for key in scores:
        scores[key] = (scores[key] / n) * 100
    
    return scores


def generation_metrics_fn(hypotheses: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute all generation metrics for a list of hypotheses and references.
    
    Args:
        hypotheses: List of generated summaries
        references: List of reference summaries
        
    Returns:
        Dictionary of all metrics
    """
    metrics = {}
    
    # Compute BLEU score
    metrics['BLEU'] = compute_bleu_score(hypotheses, references)
    
    # Compute SacreBLEU score
    metrics['SacreBLEU'] = compute_sacrebleu_score(hypotheses, references)
    
    # Compute ROUGE scores
    rouge_scores = compute_rouge_scores(hypotheses, references)
    metrics.update(rouge_scores)
    
    return metrics


def forward_generation(
    model: MBartForConditionalGeneration,
    batch_data: Tuple,
    model_type: str = 'indo-bart',
    tokenizer: IndoNLGTokenizer = None,
    device: str = 'cpu',
    is_inference: bool = False,
    is_test: bool = False,
    skip_special_tokens: bool = True,
    beam_size: int = 5,
    max_seq_len: int = 512,
    length_penalty: float = 1.0,
) -> Tuple[torch.Tensor, List[str], List[str]]:
    """
    Perform forward pass for generation.
    
    Args:
        model: Model to use for generation
        batch_data: Batch of data
        model_type: Type of model
        tokenizer: Tokenizer to use
        device: Device to use
        is_inference: Whether this is inference
        is_test: Whether this is a test
        skip_special_tokens: Whether to skip special tokens
        beam_size: Beam size for generation
        max_seq_len: Maximum sequence length
        length_penalty: Length penalty for generation
        
    Returns:
        Tuple of loss, hypotheses, and references
    """
    input_ids, attention_mask, labels, source_texts, target_texts = batch_data
    
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    
    if is_inference:
        # Generate tokens
        generated_ids = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_seq_len,
            num_beams=beam_size,
            length_penalty=length_penalty,
            early_stopping=True,
            use_cache=True,
            no_repeat_ngram_size=3
        )
        
        # Convert generated tokens to text
        generated_texts = []
        for gen_ids in generated_ids:
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=skip_special_tokens)
            generated_texts.append(gen_text)
        
        # Compute loss (even for inference to track progress)
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        loss = outputs.loss
        
        return loss, generated_texts, target_texts
    else:
        # Training or validation
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        loss = outputs.loss
        
        if is_test:
            # Generate tokens
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_seq_len,
                num_beams=beam_size,
                length_penalty=length_penalty,
                early_stopping=True,
                use_cache=True,
                no_repeat_ngram_size=3
            )
            
            # Convert generated tokens to text
            generated_texts = []
            for gen_ids in generated_ids:
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=skip_special_tokens)
                generated_texts.append(gen_text)
            
            return loss, generated_texts, target_texts
        else:
            # Return original texts for training
            return loss, source_texts, target_texts


def evaluate_model(
    model: MBartForConditionalGeneration,
    test_loader: torch.utils.data.DataLoader,
    tokenizer: IndoNLGTokenizer,
    model_type: str = 'indo-bart',
    beam_size: int = 5,
    max_seq_len: int = 512,
    device: str = 'cpu',
    output_dir: str = './results',
    length_penalty: float = 1.0,
) -> Dict[str, Any]:
    """
    Evaluate the model on the test set and save results.
    
    Args:
        model: Model to evaluate
        test_loader: DataLoader for test data
        tokenizer: Tokenizer to use
        model_type: Type of model
        beam_size: Beam size for generation
        max_seq_len: Maximum sequence length
        device: Device to use
        output_dir: Directory to save results to
        length_penalty: Length penalty for generation
        
    Returns:
        Dictionary of evaluation results
    """
    logger.info("Starting evaluation...")
    
    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Evaluate the model
    test_loss, test_metrics, test_hyp, test_label = forward_generation(
        model=model,
        batch_data=next(iter(test_loader)),
        model_type=model_type,
        tokenizer=tokenizer,
        beam_size=beam_size,
        max_seq_len=max_seq_len,
        is_inference=True,
        is_test=True,
        device=device,
        length_penalty=length_penalty
    )
    
    # Create dataframes for results
    result_df = pd.DataFrame({
        'hyp': test_hyp,
        'label': test_label
    })
    
    metrics_df = pd.DataFrame([test_metrics])
    
    # Log results
    logger.info("== Prediction Result ==")
    logger.info(result_df.head())
    logger.info("")
    
    logger.info("== Model Performance ==")
    logger.info(metrics_df.describe())
    
    # Save results
    result_path = os.path.join(output_dir, f"prediction_result_{length_penalty}.csv")
    metrics_path = os.path.join(output_dir, f"evaluation_result_{length_penalty}.csv")
    
    result_df.to_csv(result_path, index=False)
    metrics_df.describe().to_csv(metrics_path)
    
    logger.info(f"Results saved to {output_dir}")
    
    return {
        'loss': test_loss,
        'metrics': test_metrics,
        'predictions': test_hyp,
        'references': test_label,
        'result_path': result_path,
        'metrics_path': metrics_path
    }
