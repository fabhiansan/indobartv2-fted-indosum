#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Trainer Fix to debug and resolve the stuck at 0% issue with BART pretraining.
This module should be imported at the top of run_pretraining.py.

Usage:
    import trainer_fix
    trainer_fix.apply_fixes()
"""

import os
import time
import logging
import sys
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("trainer_fix.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("TrainerFix")

def patch_get_train_dataloader():
    """
    Patch the get_train_dataloader method in Trainer to add logging and
    catch potential issues with dataloader initialization.
    """
    from transformers import Trainer
    original_get_train_dataloader = Trainer.get_train_dataloader
    
    @wraps(original_get_train_dataloader)
    def patched_get_train_dataloader(self):
        logger.info("Starting to create train dataloader")
        start_time = time.time()
        
        # Set a smaller initial batch size to test loading
        original_batch_size = self.args.per_device_train_batch_size
        if original_batch_size > 2:
            logger.info(f"Temporarily reducing batch size from {original_batch_size} to 2 for testing")
            self.args.per_device_train_batch_size = 2
        
        try:
            # Create dataloader with timeout
            logger.info("Creating train dataloader with 5 minute timeout")
            
            # This is the patched part - add timeout handling
            def create_dataloader():
                return original_get_train_dataloader(self)
            
            import threading
            import queue
            
            result_queue = queue.Queue()
            
            def target_function():
                try:
                    dataloader = create_dataloader()
                    result_queue.put(("success", dataloader))
                except Exception as e:
                    result_queue.put(("error", e))
            
            thread = threading.Thread(target=target_function)
            thread.daemon = True
            thread.start()
            
            # Wait for result with timeout
            try:
                status, result = result_queue.get(timeout=300)  # 5 minutes timeout
                if status == "error":
                    raise result
                dataloader = result
            except queue.Empty:
                logger.error("Dataloader creation timed out after 5 minutes!")
                logger.info("Trying alternative dataloader creation approach...")
                
                # Reset batch size
                self.args.per_device_train_batch_size = original_batch_size
                
                # Alternative approach - simplify dataloader configuration
                from torch.utils.data import DataLoader
                dataset = self.train_dataset
                sampler = self._get_train_sampler()
                
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.args.per_device_train_batch_size,
                    sampler=sampler,
                    collate_fn=self.data_collator,
                    drop_last=self.args.dataloader_drop_last,
                    num_workers=0,  # Critical: Force single-thread
                    pin_memory=False,  # Disable pin_memory
                )
            
            # Successfully created a dataloader for testing, now restore original batch size
            if self.args.per_device_train_batch_size != original_batch_size:
                logger.info(f"Restoring original batch size to {original_batch_size}")
                self.args.per_device_train_batch_size = original_batch_size
                
                # Recreate the dataloader with the original batch size
                dataloader = original_get_train_dataloader(self)
            
            end_time = time.time()
            logger.info(f"Successfully created train dataloader in {end_time - start_time:.2f} seconds")
            
            # Test if dataloader can actually produce batches
            logger.info("Testing if dataloader can produce batches...")
            test_batch = next(iter(dataloader))
            logger.info(f"Successfully got first batch with keys: {list(test_batch.keys())}")
            
            return dataloader
            
        except Exception as e:
            end_time = time.time()
            logger.error(f"Error creating train dataloader after {end_time - start_time:.2f} seconds: {str(e)}")
            
            # Emergency fallback
            logger.warning("Using emergency fallback dataloader with minimal configuration")
            from torch.utils.data import DataLoader
            dataset = self.train_dataset
            
            # Create a very simple dataloader with minimal options
            dataloader = DataLoader(
                dataset,
                batch_size=1,  # Minimal batch size
                shuffle=True,
                collate_fn=self.data_collator,
                num_workers=0,
                pin_memory=False,
            )
            
            return dataloader
    
    # Apply the patch
    Trainer.get_train_dataloader = patched_get_train_dataloader
    logger.info("Patched Trainer.get_train_dataloader")

def patch_training_step():
    """
    Patch the training_step method in Trainer to add more diagnostic information
    and handle potential deadlocks in the training loop.
    """
    from transformers import Trainer
    original_training_step = Trainer.training_step
    
    @wraps(original_training_step)
    def patched_training_step(self, model, inputs):
        logger.info(f"Starting training step at {time.strftime('%H:%M:%S')}")
        start_time = time.time()
        
        # Log inputs shape and types for debugging
        input_debug = {}
        for k, v in inputs.items():
            if hasattr(v, "shape"):
                input_debug[k] = f"shape={v.shape}, dtype={v.dtype}"
            else:
                input_debug[k] = f"type={type(v)}"
        logger.info(f"Input batch: {input_debug}")
        
        # Apply timeout to the training step
        import threading
        import queue
        
        result_queue = queue.Queue()
        
        def target_function():
            try:
                loss = original_training_step(self, model, inputs)
                result_queue.put(("success", loss))
            except Exception as e:
                result_queue.put(("error", e))
        
        thread = threading.Thread(target=target_function)
        thread.daemon = True
        thread.start()
        
        # Wait for result with timeout (10 minutes)
        try:
            status, result = result_queue.get(timeout=600)
            if status == "error":
                logger.error(f"Error in training step: {result}")
                raise result
            loss = result
        except queue.Empty:
            logger.error("Training step timed out after 10 minutes!")
            
            # Emergency exit - this level of hanging usually indicates a serious issue
            logger.critical("Training step hung for too long, forcing process to exit")
            logger.critical("Please check your model configuration and data preprocessing")
            
            # Don't use sys.exit() as it might be caught by exception handlers
            os._exit(1)
        
        end_time = time.time()
        logger.info(f"Completed training step in {end_time - start_time:.2f} seconds, loss: {loss}")
        
        return loss
    
    # Apply the patch
    Trainer.training_step = patched_training_step
    logger.info("Patched Trainer.training_step")

def patch_bart_modeling():
    """
    Patch BART model to address potential issues in the forward pass
    that might cause it to hang.
    """
    try:
        from transformers.models.bart.modeling_bart import BartForConditionalGeneration
        
        original_forward = BartForConditionalGeneration.forward
        
        @wraps(original_forward)
        def patched_forward(self, *args, **kwargs):
            start_time = time.time()
            logger.info("Starting BART forward pass")
            
            # Run with timeout
            import threading
            import queue
            
            result_queue = queue.Queue()
            
            def target_function():
                try:
                    output = original_forward(self, *args, **kwargs)
                    result_queue.put(("success", output))
                except Exception as e:
                    result_queue.put(("error", e))
            
            thread = threading.Thread(target=target_function)
            thread.daemon = True
            thread.start()
            
            # Wait for result with timeout (5 minutes)
            try:
                status, result = result_queue.get(timeout=300)
                if status == "error":
                    logger.error(f"Error in BART forward pass: {result}")
                    raise result
                output = result
            except queue.Empty:
                logger.error("BART forward pass timed out after 5 minutes!")
                logger.error("This likely indicates an issue with the model configuration or inputs")
                
                # Emergency exit
                logger.critical("BART forward pass hung for too long, forcing process to exit")
                os._exit(1)
            
            end_time = time.time()
            logger.info(f"Completed BART forward pass in {end_time - start_time:.2f} seconds")
            
            return output
        
        # Apply the patch
        BartForConditionalGeneration.forward = patched_forward
        logger.info("Patched BartForConditionalGeneration.forward")
        
    except ImportError:
        logger.warning("Could not patch BART model, module not found")

def apply_fixes():
    """Apply all fixes to resolve the training stuck issue."""
    logger.info("Applying trainer fixes...")
    
    # Critical: Disable wandb which can cause hanging
    os.environ["WANDB_DISABLED"] = "true"
    logger.info("Disabled wandb via environment variable")
    
    # Optimize for better performance
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    logger.info("Disabled tokenizers parallelism")
    
    # Set environment variables for debugging
    os.environ["PYTHONUNBUFFERED"] = "1"
    logger.info("Set Python unbuffered mode for better logging")
    
    # Apply patches
    patch_get_train_dataloader()
    patch_training_step()
    patch_bart_modeling()
    
    logger.info("All trainer fixes applied successfully")

# Automatically apply fixes when imported
if __name__ != "__main__":
    apply_fixes()
