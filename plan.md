# IndoBART-v2 Fine-tuning on IndoSUM Dataset

## Project Plan

This document outlines the development process for fine-tuning an IndoBART-v2 model on the IndoSUM dataset and pushing the fine-tuned model to Huggingface Hub.

### File Structure

1. **data_loader.py**
   - Contains dataset handling logic for IndoSUM dataset
   - Implements custom Dataset and DataLoader classes
   - Location references: `../dataset/indosum`

2. **train.py**
   - Training loop implementation
   - Learning rate scheduling
   - Early stopping mechanism
   - Checkpoint saving

3. **evaluate.py**
   - Evaluation functionality
   - Metrics calculation (BLEU, SacreBLEU, ROUGE scores)
   - Generation of prediction results

4. **hub_utils.py**
   - Huggingface Hub integration
   - Model and tokenizer pushing functionality

5. **main.py**
   - Orchestrates the entire process
   - Argument parsing
   - Model loading and configuration
   - Training, evaluation, and pushing

### Progress Tracking

- [ ] Create data_loader.py
- [ ] Create train.py
- [ ] Create evaluate.py
- [ ] Create hub_utils.py
- [ ] Create main.py
- [ ] Test full pipeline locally
- [ ] Push to Huggingface Hub

### Implementation Notes

#### data_loader.py
- Implement SummarizationDataset class
- Create SummarizationDataLoader

#### train.py
- Implement train() function
- Add checkpoint saving functionality
- Add learning rate scheduling

#### evaluate.py
- Implement evaluate() function
- Add metrics computation (BLEU, SacreBLEU, ROUGE)
- Add prediction output functionality

#### hub_utils.py
- Implement push_to_hub() function

#### main.py
- Parse command line arguments
- Create training and evaluation pipelines
- Handle model loading and saving
- Push to Huggingface Hub

### Debugging Reference

#### Common Issues
- IndoNLGTokenizer might need special handling for language IDs
- Sequence length limitations (512 tokens)
- Memory management for large batch sizes
- GPU CUDA issues

#### Helpful Commands
- Check CUDA availability: `torch.cuda.is_available()`
- Memory usage: `torch.cuda.memory_summary()`
- Model parameter count: `sum(p.numel() for p in model.parameters())`
