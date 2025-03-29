"""
Data loading utilities for the IndoSUM dataset.
"""
from typing import Dict, List, Optional, Tuple, Union, Any

import torch
from torch.utils.data import Dataset, DataLoader
import datasets
import os
from transformers import PreTrainedTokenizerBase
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
        tokenizer: PreTrainedTokenizerBase,
        max_source_length: int = 1024,
        max_target_length: int = 256,
        lowercase: bool = True,
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
            swap_source_target: Whether to swap source and target
        """
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.lowercase = lowercase
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

        # Check if the dataset directory exists
        if not os.path.exists(dataset_path):
            logger.error(f"Dataset directory does not exist: {dataset_path}")
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        
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

            # Encode source and target using standard tokenizer
            # The tokenizer will handle adding special tokens based on its configuration
            model_inputs = self.tokenizer(
                source_text,
                max_length=self.max_source_length,
                padding=False, # Don't pad source here, handle in dataloader if needed
                truncation=True, # Truncate source if longer than max_source_length
                return_tensors=None, # Get list of IDs first
            )

            # Encode target for labels
            labels = self.tokenizer(
                text_target=target_text, # Use text_target for labels
                max_length=self.max_target_length,
                padding=False, # Don't pad target here
                truncation=True, # Truncate target if longer than max_target_length
                return_tensors=None, # Get list of IDs first
            ).input_ids # Get only input_ids for labels

            # Convert to tensors
            input_ids = torch.tensor(model_inputs['input_ids'])
            attention_mask = torch.tensor(model_inputs['attention_mask'])
            labels = torch.tensor(labels)

            # Replace padding token id in labels with -100 for CrossEntropyLoss
            # This should happen *after* batching and padding in the collate_fn usually.
            # However, if SummarizationDataLoader doesn't handle this, we do it here assuming no padding yet.
            # Revisit this if using padding='max_length' or a custom collate_fn.
            # labels[labels == self.tokenizer.pad_token_id] = -100 # Move this logic to collate_fn if possible

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels, # Labels are now target_ids
                "source_text": source_text, # Keep original text if needed downstream
                "target_text": target_text,
            }
        except Exception as e:
            logger.error(f"Error processing example at index {idx}: {str(e)}")
            raise


class SummarizationDataLoader(DataLoader):
    """
    DataLoader for summarization datasets.
    
    This DataLoader handles collation of examples into batches for training
    and evaluation of summarization models. Includes dynamic padding.
    """
    
    def __init__(
        self,
        dataset: IndoSUMDataset,
        tokenizer: PreTrainedTokenizerBase,
        batch_size: int = 8,
        num_workers: int = 2,  
        shuffle: bool = True,
        **kwargs # Allow passing other DataLoader args
    ):
        """
        Initialize the DataLoader.
        
        Args:
            dataset: Dataset to load data from
            tokenizer: Tokenizer used for padding (needed in collate_fn)
            batch_size: Batch size
            num_workers: Number of workers to use for loading data
            shuffle: Whether to shuffle the data
        """
        self.dataset = dataset
        self.tokenizer = tokenizer # Keep tokenizer for collate_fn
        
        logger.info(f"Initializing SummarizationDataLoader with {num_workers} workers")
        
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=self.collate_fn, # Use custom collate_fn
            **kwargs # Pass other args like pin_memory
        )

    def collate_fn(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate function to pad sequences dynamically within a batch.
        Replaces padding token IDs in labels with -100.
        """
        # Extract components from the batch
        input_ids = [item['input_ids'] for item in batch]
        attention_mask = [item['attention_mask'] for item in batch]
        labels = [item['labels'] for item in batch]
        source_texts = [item['source_text'] for item in batch] # Keep original text
        target_texts = [item['target_text'] for item in batch] # Keep original text

        # Pad input_ids and attention_mask
        padded_inputs = self.tokenizer.pad(
            {'input_ids': input_ids},
            padding='longest', # Pad to the longest sequence in the batch
            return_tensors='pt'
        )

        # Pad labels separately
        padded_labels = self.tokenizer.pad(
            {'input_ids': labels}, # Treat labels like input_ids for padding
            padding='longest',
            return_tensors='pt'
        ).input_ids # Get the padded IDs tensor

        # Replace padding token ID in labels with -100
        padded_labels[padded_labels == self.tokenizer.pad_token_id] = -100

        # Return the collated batch
        return {
            'input_ids': padded_inputs['input_ids'],
            'attention_mask': padded_inputs['attention_mask'],
            'labels': padded_labels,
            'source_text': source_texts, # Include original texts if needed later
            'target_text': target_texts,
        }
