"""
Evaluation functionality for IndoBART fine-tuned on the IndoSUM dataset.
"""
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
import time

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
from bert_score import score

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
    
    # Compute score with force=True to suppress tokenization warnings
    # since we're handling tokenization separately in the tokenizer
    bleu = sacrebleu.corpus_bleu(hypotheses, refs, force=True)
    
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
        # Calculate scores for this pair
        score = scorer.score(ref, hyp)
        
        # Add precision scores
        scores['ROUGE1'] += score['rouge1'].fmeasure
        scores['ROUGE2'] += score['rouge2'].fmeasure
        scores['ROUGEL'] += score['rougeL'].fmeasure
        scores['ROUGELsum'] += score['rougeLsum'].fmeasure
    
    # Calculate averages
    num_pairs = len(hypotheses)
    for key in scores:
        scores[key] = (scores[key] / num_pairs) * 100
    
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
    # Compute BLEU score
    bleu = compute_bleu_score(hypotheses, references)
    
    # Compute SacreBLEU score
    sacrebleu_score = compute_sacrebleu_score(hypotheses, references)
    
    # Compute ROUGE scores
    rouge_scores = compute_rouge_scores(hypotheses, references)

    # Compute BERTScore
    P, R, F1 = score(hypotheses, references, lang="id", verbose=True)
    bertscore_f1 = F1.mean().item()
    bertscore_precision = P.mean().item()
    bertscore_recall = R.mean().item()

    # Combine all metrics
    metrics = {
        'BLEU': bleu,
        'SacreBLEU': sacrebleu_score,
        **rouge_scores,
        'BERTScore_F1': bertscore_f1,
        'BERTScore_Precision': bertscore_precision,
        'BERTScore_Recall': bertscore_recall,
    }

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
    start_time = time.time()
    logger.info(f"[{time.strftime('%H:%M:%S')}] Forward generation started (is_inference={is_inference}, is_test={is_test})")
    
    input_ids, attention_mask, labels, source_texts, target_texts = batch_data
    
    # Log batch size and sequence length
    batch_size = input_ids.size(0)
    seq_length = input_ids.size(1)
    logger.info(f"Batch size: {batch_size}, Sequence length: {seq_length}")
    
    # Move tensors to device
    tensor_start = time.time()
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    logger.info(f"Moving tensors to device took {time.time() - tensor_start:.2f}s")
    
    if is_inference:
        # Generate tokens
        logger.info(f"[{time.strftime('%H:%M:%S')}] Starting token generation with beam_size={beam_size}")
        if torch.cuda.is_available():
            logger.info(f"GPU memory before generation: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        gen_start = time.time()
        try:
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
            gen_time = time.time() - gen_start
            logger.info(f"[{time.strftime('%H:%M:%S')}] Generation completed in {gen_time:.2f}s ({gen_time/batch_size:.2f}s per example)")
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            logger.exception("Exception details:")
            raise
        
        if torch.cuda.is_available():
            logger.info(f"GPU memory after generation: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        # Convert generated tokens to text
        decode_start = time.time()
        generated_texts = []
        for gen_ids in generated_ids:
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=skip_special_tokens)
            generated_texts.append(gen_text)
        logger.info(f"Decoding took {time.time() - decode_start:.2f}s")
        
        # Compute loss (even for inference to track progress)
        loss_start = time.time()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        loss = outputs.loss
        logger.info(f"Loss computation took {time.time() - loss_start:.2f}s")
        
        logger.info(f"[{time.strftime('%H:%M:%S')}] Forward generation completed in {time.time() - start_time:.2f}s")
        return loss, generated_texts, target_texts
    else:
        # Training or validation
        loss_start = time.time()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        loss = outputs.loss
        logger.info(f"Loss computation took {time.time() - loss_start:.2f}s")
        
        if is_test:
            # Generate tokens
            logger.info(f"[{time.strftime('%H:%M:%S')}] Starting token generation for test")
            gen_start = time.time()
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
            logger.info(f"Generation took {time.time() - gen_start:.2f}s")
            
            # Convert generated tokens to text
            decode_start = time.time()
            generated_texts = []
            for gen_ids in generated_ids:
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=skip_special_tokens)
                generated_texts.append(gen_text)
            logger.info(f"Decoding took {time.time() - decode_start:.2f}s")
            
            logger.info(f"[{time.strftime('%H:%M:%S')}] Forward generation for test completed in {time.time() - start_time:.2f}s")
            return loss, generated_texts, target_texts
        else:
            # Return original texts for training
            logger.info(f"[{time.strftime('%H:%M:%S')}] Forward pass for training completed in {time.time() - start_time:.2f}s")
            return loss, source_texts, target_texts


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
    logger.info(f"Beginning evaluation with model_type={model_type}, beam_size={beam_size}, max_seq_len={max_seq_len}")
    logger.info(f"Tokenizer details: {tokenizer.__class__.__name__}, vocab_size={tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 'N/A'}")
    
    model.eval()
    torch.set_grad_enabled(False)
    
    if torch.cuda.is_available():
        logger.info(f"GPU memory before evaluation: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    total_loss = 0
    list_hyp = []
    list_label = []
    
    start_time = time.time()
    logger.info(f"[{time.strftime('%H:%M:%S')}] Progress bar created, starting evaluation loop...")
    pbar = tqdm(iter(data_loader), leave=True, total=len(data_loader))
    
    for i, batch_data in enumerate(pbar):
        # Log every 500 batches or at 25%, 50%, 75% points
        should_log = (i % 500 == 0) or any(i == int(len(data_loader) * p) for p in [0.25, 0.5, 0.75])
        if should_log:
            logger.info(f"[{time.strftime('%H:%M:%S')}] Processing batch {i}/{len(data_loader)} ({i/len(data_loader)*100:.1f}%)")
            if torch.cuda.is_available():
                logger.info(f"GPU memory during evaluation: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
        try:
            batch_start = time.time()
            loss, batch_hyp, batch_label = forward_fn(
                model, batch_data, model_type=model_type, tokenizer=tokenizer,
                is_inference=True, device=device, skip_special_tokens=True,
                beam_size=beam_size, max_seq_len=max_seq_len, length_penalty=length_penalty
            )
            
            if should_log and i > 0:
                batch_time = time.time() - batch_start
                estimated_remaining = batch_time * (len(data_loader) - i)
                logger.info(f"Batch processing time: {batch_time:.2f}s, Estimated remaining: {estimated_remaining/60:.2f} minutes")
            
            val_loss = loss.item()
            total_loss = total_loss + val_loss
            
            # Store hypotheses and references
            list_hyp += batch_hyp
            list_label += batch_label
            
            pbar.set_description("VALID LOSS:%.4f" % (total_loss/(i+1)))
        except Exception as e:
            logger.error(f"Error processing batch {i}: {str(e)}")
            logger.exception("Exception details:")
    
    eval_time = time.time() - start_time
    logger.info(f"[{time.strftime('%H:%M:%S')}] Evaluation loop completed in {eval_time:.2f}s ({eval_time/60:.2f} minutes)")
    
    # Calculate metrics
    logger.info(f"[{time.strftime('%H:%M:%S')}] Starting metrics calculation for {len(list_hyp)} examples...")
    metrics_start = time.time()
    
    # Log sample predictions for debugging
    for i in range(min(3, len(list_hyp))):
        logger.info(f"Sample {i}: \nReference: {list_label[i][:100]}... \nHypothesis: {list_hyp[i][:100]}...")
    
    if torch.cuda.is_available():
        logger.info(f"GPU memory before metrics calculation: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        
    try:
        # Calculate BLEU first and log
        logger.info(f"[{time.strftime('%H:%M:%S')}] Calculating BLEU score...")
        bleu_start = time.time()
        bleu = compute_bleu_score(list_hyp, list_label)
        logger.info(f"BLEU calculation took {time.time() - bleu_start:.2f}s, Score: {bleu:.2f}")
        
        # Then SacreBLEU
        logger.info(f"[{time.strftime('%H:%M:%S')}] Calculating SacreBLEU score...")
        sacrebleu_start = time.time()
        sacrebleu_score = compute_sacrebleu_score(list_hyp, list_label)
        logger.info(f"SacreBLEU calculation took {time.time() - sacrebleu_start:.2f}s, Score: {sacrebleu_score:.2f}")
        
        # Finally ROUGE scores
        logger.info(f"[{time.strftime('%H:%M:%S')}] Calculating ROUGE scores...")
        rouge_start = time.time()
        rouge_scores = compute_rouge_scores(list_hyp, list_label)
        logger.info(f"ROUGE calculation took {time.time() - rouge_start:.2f}s, Scores: {rouge_scores}")
        
        # Combine all metrics
        metrics = {
            'BLEU': bleu,
            'SacreBLEU': sacrebleu_score,
            **rouge_scores
        }
    except Exception as e:
        logger.error(f"Error calculating metrics: {str(e)}")
        logger.exception("Exception details:")
        metrics = {'ERROR': str(e)}
    
    metrics_time = time.time() - metrics_start
    logger.info(f"[{time.strftime('%H:%M:%S')}] Metrics calculation completed in {metrics_time:.2f}s ({metrics_time/60:.2f} minutes)")
    
    # Calculate average loss
    avg_loss = total_loss / len(data_loader)
    logger.info(f"[{time.strftime('%H:%M:%S')}] Final average loss: {avg_loss:.4f}")
    
    if torch.cuda.is_available():
        logger.info(f"GPU memory after evaluation: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    
    # Return results
    logger.info(f"[{time.strftime('%H:%M:%S')}] Returning results from evaluate function")
    if is_test:
        return avg_loss, metrics, list_hyp, list_label
    else:
        return avg_loss, metrics


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
    test_loss, test_metrics, test_hyp, test_label = evaluate(
        model=model,
        data_loader=test_loader,
        forward_fn=forward_generation,
        metrics_fn=generation_metrics_fn,
        model_type=model_type,
        tokenizer=tokenizer,
        beam_size=beam_size,
        max_seq_len=max_seq_len,
        is_test=True,
        device=device,
        length_penalty=length_penalty,
    )
    
    # Print metrics
    logger.info(f'Test loss: {test_loss:.4f}')
    for metric_name, metric_value in test_metrics.items():
        logger.info(f'Test {metric_name}: {metric_value:.2f}')
    
    # Save results in CSV format
    results = []
    for i, (hyp, ref) in enumerate(zip(test_hyp, test_label)):
        results.append({
            'id': i,
            'reference': ref,
            'generated': hyp
        })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, 'test_results.csv'), index=False)
    
    # Calculate metrics on saved results as a sanity check
    verify_metrics = generation_metrics_fn(df['generated'].tolist(), df['reference'].tolist())
    logger.info('Metrics on saved results:')
    for metric_name, metric_value in verify_metrics.items():
        logger.info(f'Verified {metric_name}: {metric_value:.2f}')
    
    # Return evaluation results
    return {
        'loss': test_loss,
        'metrics': test_metrics,
        'hypotheses': test_hyp,
        'references': test_label
    }
