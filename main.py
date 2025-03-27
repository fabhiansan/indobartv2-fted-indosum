"""
Main script for fine-tuning IndoBART-v2 on the IndoSUM dataset.
"""
from typing import Dict, List, Optional, Tuple, Union, Any
import os
import sys
import logging
import argparse
import torch
import numpy as np
import random
from pathlib import Path

# Import local modules
from data_loader import IndoSUMDataset, SummarizationDataLoader
from train import train, set_seed
from evaluate import forward_generation, generation_metrics_fn, evaluate_model
from hub_utils import save_model_to_disk, push_to_hub, create_model_card

# Import transformers
from transformers import MBartForConditionalGeneration
from indonlg.modules.tokenization_indonlg import IndoNLGTokenizer
import transformers

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

# Set transformers logging level to INFO to see more details
transformers.logging.set_verbosity_info()

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Fine-tune IndoBART on IndoSUM dataset")
    
    # Data arguments
    parser.add_argument("--data_dir", type=str, default="../dataset/indosum",
                        help="Directory containing the dataset")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory to save outputs")
    parser.add_argument("--model_dir", type=str, default="./save",
                        help="Directory to save models")
    parser.add_argument("--cache_dir", type=str, default="./cache",
                        help="Directory to cache models and tokenizers")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="indobenchmark/indobart",
                        help="Pretrained model name or path")
    parser.add_argument("--model_type", type=str, default="indo-bart",
                        help="Model type (indo-bart)")
    parser.add_argument("--max_seq_len", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--max_source_length", type=int, default=1024,
                        help="Maximum source sequence length")
    parser.add_argument("--max_target_length", type=int, default=256,
                        help="Maximum target sequence length")
    
    # Training arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--grad_accum", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=3e-5,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay")
    parser.add_argument("--max_norm", type=float, default=1.0,
                        help="Maximum gradient norm")
    parser.add_argument("--n_epochs", type=int, default=5,
                        help="Number of epochs")
    parser.add_argument("--evaluate_every", type=int, default=1,
                        help="Evaluate every n epochs")
    parser.add_argument("--early_stop", type=int, default=3,
                        help="Early stopping patience")
    parser.add_argument("--valid_criterion", type=str, default="ROUGE1",
                        help="Validation criterion")
    parser.add_argument("--beam_size", type=int, default=5,
                        help="Beam size for generation")
    parser.add_argument("--length_penalty", type=float, default=1.0,
                        help="Length penalty for generation")
    parser.add_argument("--fp16", action="store_true",
                        help="Use FP16 precision")
    
    # Dataloader arguments
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of workers for dataloaders")
    
    # Hugging Face Hub arguments
    parser.add_argument("--push_to_hub", action="store_true",
                        help="Push model to Hugging Face Hub")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Repository ID for Hugging Face Hub")
    parser.add_argument("--hub_private", action="store_true",
                        help="Make Hugging Face Hub repository private")
    parser.add_argument("--hub_token", type=str, default=None,
                        help="Hugging Face Hub token")
    
    # Experiment arguments
    parser.add_argument("--exp_id", type=str, default=None,
                        help="Experiment ID")
    
    return parser.parse_args()


def checkpoint_callback(
    model: torch.nn.Module,
    epoch: int,
    checkpoint_path: str,
    metric_value: float,
) -> None:
    """
    Callback for saving checkpoints.
    
    Args:
        model: Model to save
        epoch: Current epoch
        checkpoint_path: Path to save checkpoint to
        metric_value: Value of the validation metric
    """
    try:
        logger.info(f"Saving checkpoint at epoch {epoch+1} with {metric_value:.4f}...")
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {str(e)}")
    

def main() -> None:
    """
    Main function for fine-tuning IndoBART on IndoSUM dataset.
    """
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.cache_dir, exist_ok=True)
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info("CUDA is available, using GPU.")
    else:
        logger.info("CUDA is not available, using CPU.")
    logger.info(f"Using device: {device}")

    # Set experiment ID
    if args.exp_id is None:
        args.exp_id = f"indosum_{args.model_type}_bs{args.batch_size}_lr{args.lr}_seed{args.seed}"
        logger.info(f"Experiment ID: {args.exp_id}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer from {args.model_name}")
    try:
        # Add more detailed logging for tokenizer initialization
        logger.info("Tokenizer initialization details:")
        logger.info(f"  - Model name: {args.model_name}")
        logger.info(f"  - Cache directory: {args.cache_dir}")
        
        tokenizer = IndoNLGTokenizer.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        logger.info(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
        logger.info(f"Tokenizer vocabulary size: {tokenizer.vocab_size if hasattr(tokenizer, 'vocab_size') else 'N/A'}")
        logger.info(f"Tokenizer special tokens: {tokenizer.all_special_tokens}")
        logger.info(f"Tokenizer special token ids: {[tokenizer.convert_tokens_to_ids(t) for t in tokenizer.all_special_tokens]}")
        
        # Log tokenizer attributes to help debug the 'default tokenizer' message
        logger.info("Tokenizer Configuration Details:")
        for attr_name in dir(tokenizer):
            if not attr_name.startswith('_') and not callable(getattr(tokenizer, attr_name)):
                try:
                    attr_value = getattr(tokenizer, attr_name)
                    if not isinstance(attr_value, (dict, list)) or len(str(attr_value)) < 100:
                        logger.info(f"  - {attr_name}: {attr_value}")
                except Exception:
                    pass
        
        logger.info(f"Tokenizer successfully loaded")
        logger.info(f"Vocabulary size: {len(tokenizer)}")
        logger.info(f"Tokenizer type: {type(tokenizer).__name__}")

        # Check tokenizer compatibility
        if not hasattr(tokenizer, 'prepare_input_for_generation'):
            logger.error("Tokenizer does not have 'prepare_input_for_generation' method. Incompatible tokenizer.")
            raise ValueError("Incompatible tokenizer: Tokenizer must have 'prepare_input_for_generation' method.")

        # Log special tokens
        special_tokens = {
            "Padding token": tokenizer.pad_token,
            "Unknown token": tokenizer.unk_token,
            "SOS/BOS token": tokenizer.bos_token,
            "EOS token": tokenizer.eos_token
        }
        logger.info("Special tokens:")
        for name, token in special_tokens.items():
            if token:
                logger.info(f"  - {name}: '{token}' (id: {tokenizer.convert_tokens_to_ids(token)})")

    except Exception as e:
        logger.error(f"Failed to load tokenizer: {str(e)}")
        raise RuntimeError(f"Tokenizer loading failed: {str(e)}")

    # Load model
    logger.info(f"Loading model from {args.model_name}")
    try:
        model = MBartForConditionalGeneration.from_pretrained(args.model_name, cache_dir=args.cache_dir)
        model = model.to(device)
        logger.info(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")
    
    # Validate data directory exists
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory does not exist: {args.data_dir}")
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")
    
    # Load datasets
    logger.info("Loading datasets")
    try:
        train_dataset = IndoSUMDataset(
            data_dir=args.data_dir,
            split="train",
            tokenizer=tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            source_lang="[indonesian]",
            target_lang="[indonesian]"
        )
    
        valid_dataset = IndoSUMDataset(
            data_dir=args.data_dir,
            split="dev",
            tokenizer=tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            source_lang="[indonesian]",
            target_lang="[indonesian]"
        )
    
        test_dataset = IndoSUMDataset(
            data_dir=args.data_dir,
            split="test",
            tokenizer=tokenizer,
            max_source_length=args.max_source_length,
            max_target_length=args.max_target_length,
            source_lang="[indonesian]",
            target_lang="[indonesian]"
        )
        
        logger.info(f"Loaded {len(train_dataset)} training examples")
        logger.info(f"Loaded {len(valid_dataset)} validation examples")
        logger.info(f"Loaded {len(test_dataset)} test examples")
    except Exception as e:
        logger.error(f"Failed to load datasets: {str(e)}")
        raise RuntimeError(f"Dataset loading failed: {str(e)}")
    
    # Create data loaders
    logger.info("Creating data loaders")
    
    # Use the num_workers from args
    train_loader = SummarizationDataLoader(
        dataset=train_dataset,
        model_type=args.model_type,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    valid_loader = SummarizationDataLoader(
        dataset=valid_dataset,
        model_type=args.model_type,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    test_loader = SummarizationDataLoader(
        dataset=test_dataset,
        model_type=args.model_type,
        tokenizer=tokenizer,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Train model
    logger.info("Starting training")
    history = train(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        optimizer=optimizer,
        forward_fn=forward_generation,
        metrics_fn=generation_metrics_fn,
        valid_criterion=args.valid_criterion,
        tokenizer=tokenizer,
        n_epochs=args.n_epochs,
        evaluate_every=args.evaluate_every,
        early_stop=args.early_stop,
        step_size=1,
        gamma=0.5,
        max_norm=args.max_norm,
        grad_accum=args.grad_accum,
        beam_size=args.beam_size,
        max_seq_len=args.max_seq_len,
        model_type=args.model_type,
        model_dir=args.model_dir,
        exp_id=args.exp_id,
        fp16=args.fp16,
        device=device,
        checkpoint_callback=checkpoint_callback,
    )
    
    # Load best model
    best_model_path = os.path.join(args.model_dir, f"best_model_{args.exp_id}.pt")
    if os.path.exists(best_model_path):
        try:
            logger.info(f"Loading best model from {best_model_path}")
            model.load_state_dict(torch.load(best_model_path))
            logger.info("Successfully loaded best model")
        except Exception as e:
            logger.error(f"Failed to load best model: {str(e)}")
            logger.warning("Continuing with current model weights")
    
    # Evaluate model
    logger.info("Evaluating model on test set")
    evaluation_results = evaluate_model(
        model=model,
        test_loader=test_loader,
        tokenizer=tokenizer,
        model_type=args.model_type,
        beam_size=args.beam_size,
        max_seq_len=args.max_seq_len,
        device=device,
        output_dir=args.output_dir,
        length_penalty=args.length_penalty,
    )
    
    # Save model
    if not args.push_to_hub:
        logger.info(f"Saving model to {args.model_dir}/{args.exp_id}")
        save_model_to_disk(
            model=model,
            tokenizer=tokenizer,
            output_dir=os.path.join(args.model_dir, args.exp_id)
        )
    
    # Push model to Hugging Face Hub
    if args.push_to_hub:
        logger.info("Pushing model to Hugging Face Hub")
        
        if not args.hub_token:
            logger.error("Hugging Face Hub token is required when push_to_hub is enabled")
            raise ValueError("Hugging Face Hub token is required")
            
        if args.hub_model_id is None:
            args.hub_model_id = f"indobenchmark/indobart-finetuned-indosum-{args.exp_id}"
        
        # Create model card
        model_card = create_model_card(
            repo_id=args.hub_model_id,
            metrics=evaluation_results['metrics'],
            model_name="IndoBART-v2",
            dataset_name="IndoSUM",
            language="Indonesian",
            license_name="MIT",
            finetuned_from="indobenchmark/indobart",
            tasks=["summarization"],
            output_file=os.path.join(args.output_dir, "README.md")
        )
        
        # Push to hub
        try:
            push_to_hub(
                model=model,
                tokenizer=tokenizer,
                repo_id=args.hub_model_id,
                private=args.hub_private,
                token=args.hub_token,
                local_dir=os.path.join(args.model_dir, args.exp_id)
            )
            logger.info(f"Model successfully pushed to {args.hub_model_id}")
        except Exception as e:
            logger.error(f"Failed to push model to Hugging Face Hub: {str(e)}")
            logger.warning("Model was not pushed to Hub - saving locally instead")
            save_model_to_disk(
                model=model,
                tokenizer=tokenizer,
                output_dir=os.path.join(args.model_dir, args.exp_id)
            )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
