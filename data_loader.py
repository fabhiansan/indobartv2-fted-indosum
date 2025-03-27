"""
Data loading utilities for the IndoSUM dataset.
"""
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch.utils.data import Dataset, DataLoader
import datasets
import os
from transformers import PreTrainedTokenizer
import logging

logger = logging.getLogger(__name__)


class IndoSUMDataset(Dataset):
    """
    Dataset for Indonesian summarization tasks using IndoBART.
    
    This dataset handles loading and preprocessing data from the IndoSUM dataset,
    transforming it into a format suitable for training a summarization model.
    """
    
    def __init__(
        self,
        data_dir: str,
        split: str,
        tokenizer: PreTrainedTokenizer,
        max_source_length: int = 1024,
        max_target_length: int = 256,
        lowercase: bool = True,
        no_special_token: bool = False,
        source_lang: str = "[indonesian]",
        target_lang: str = "[indonesian]",
        swap_source_target: bool = False,
        **kwargs
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the dataset
            split: Split to use (train, dev, test)
            tokenizer: Tokenizer to use for preprocessing
            max_source_length: Maximum length of the source text
            max_target_length: Maximum length of the target text
            lowercase: Whether to lowercase the text
            no_special_token: Whether to include special tokens
            source_lang: Source language token
            target_lang: Target language token
            swap_source_target: Whether to swap source and target
        """
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.lowercase = lowercase
        self.no_special_token = no_special_token
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.swap_source_target = swap_source_target
        
        # Map split names to dataset directory names
        split_map = {
            'train': 'traindataset',
            'dev': 'devdataset',
            'valid': 'devdataset',
            'test': 'testdataset',
        }
        
        if split not in split_map:
            raise ValueError(f"Split {split} not recognized. Must be one of {list(split_map.keys())}.")
        
        dataset_path = os.path.join(data_dir, split_map[split])
        
        logger.info(f"Loading IndoSUM dataset from {dataset_path}")
        try:
            self.dataset = datasets.load_from_disk(dataset_path)
            logger.info(f"Successfully loaded dataset with {len(self.dataset)} examples")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
        
        # Log dataset information
        logger.info(f"Dataset columns: {self.dataset.column_names}")
    
    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single example from the dataset.
        
        Args:
            idx: Index of the example to get
            
        Returns:
            Dictionary containing input_ids, attention_mask, and labels
        """
        try:
            example = self.dataset[idx]
            
            # Get source and target text
            source_text = example['article']
            target_text = example['summary']
            
            # Apply preprocessing
            if self.lowercase:
                source_text = source_text.lower()
                target_text = target_text.lower()
            
            # Swap source and target if needed
            if self.swap_source_target:
                source_text, target_text = target_text, source_text
            
            # Encode the source text
            source_encoded = self.tokenizer.prepare_input_for_generation(
                [source_text],
                return_tensors='pt',
                lang_token=self.source_lang,
                decoder_lang_token=self.target_lang,
                max_length=self.max_source_length,
                truncation=True,
            )
            
            # Encode the target text
            target_encoding = self.tokenizer(
                target_text,
                max_length=self.max_target_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            
            # Get the encoded ids
            input_ids = source_encoded["input_ids"].squeeze(0)
            attention_mask = source_encoded["attention_mask"].squeeze(0)
            labels = target_encoding["input_ids"].squeeze(0)
            
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "source_text": source_text,
                "target_text": target_text,
            }
        except Exception as e:
            logger.error(f"Error processing example at index {idx}: {str(e)}")
            raise


class SummarizationDataLoader(DataLoader):
    """
    DataLoader for summarization datasets.
    
    This DataLoader handles collation of examples into batches for training
    and evaluation of summarization models.
    """
    
    def __init__(
        self,
        dataset: IndoSUMDataset,
        model_type: str,
        tokenizer: PreTrainedTokenizer,
        max_seq_len: int = 512,
        batch_size: int = 8,
        src_lid_token_id: int = -1,
        tgt_lid_token_id: int = -1,
        num_workers: int = 2,  
        shuffle: bool = True,
    ):
        """
        Initialize the DataLoader.
        
        Args:
            dataset: Dataset to load data from
            model_type: Type of model being used
            tokenizer: Tokenizer to use for preprocessing
            max_seq_len: Maximum sequence length
            batch_size: Batch size
            src_lid_token_id: Source language ID token
            tgt_lid_token_id: Target language ID token
            num_workers: Number of workers to use for loading data
            shuffle: Whether to shuffle the data
        """
        self.dataset = dataset
        self.model_type = model_type
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.src_lid_token_id = src_lid_token_id
        self.tgt_lid_token_id = tgt_lid_token_id
        
        logger.info(f"Initializing SummarizationDataLoader with {num_workers} workers")
        
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            timeout=60,  
            pin_memory=torch.cuda.is_available(),  
            prefetch_factor=2 if num_workers > 0 else None,  
        )
        
        logger.info(f"SummarizationDataLoader initialized with {len(dataset)} examples")
    
    def _collate_fn(self, examples: List[Dict[str, torch.Tensor]]) -> Tuple:
        """
        Collate examples into a batch.
        
        Args:
            examples: List of examples to collate
            
        Returns:
            Tuple of tensors for input_ids, attention_mask, and labels
        """
        try:
            # Stack input_ids, attention_mask, and labels
            input_ids = torch.stack([ex['input_ids'] for ex in examples])
            attention_mask = torch.stack([ex['attention_mask'] for ex in examples])
            labels = torch.stack([ex['labels'] for ex in examples])
            
            # Get source and target text
            source_text = [ex['source_text'] for ex in examples]
            target_text = [ex['target_text'] for ex in examples]
            
            return (input_ids, attention_mask, labels, source_text, target_text)
        except Exception as e:
            # Log detailed information about the exception and examples
            logger.error(f"Error in _collate_fn: {str(e)}")
            for i, ex in enumerate(examples):
                logger.error(f"Example {i} keys: {ex.keys()}")
            raise
