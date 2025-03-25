# IndoNLG Documentation

## Overview
IndoNLG is a comprehensive collection of Natural Language Generation (NLG) resources for Bahasa Indonesia. It includes pre-trained models, datasets, and code for various downstream NLP tasks in Indonesian. The project is a collaboration between various universities and industry partners including Institut Teknologi Bandung, Universitas Multimedia Nusantara, The Hong Kong University of Science and Technology, Universitas Indonesia, DeepMind, Gojek, and Prosa.AI.

## Key Components

### Pre-trained Models
1. **IndoBART** - A BART-based sequence-to-sequence model for Indonesian
   - Available on HuggingFace: [indobenchmark/indobart](https://huggingface.co/indobenchmark/indobart)
   - Architecture: Based on the BART architecture with encoder-decoder transformer

2. **IndoBART-v2** - An improved version of IndoBART
   - Available on HuggingFace: [indobenchmark/indobart-v2](https://huggingface.co/indobenchmark/indobart-v2)

3. **IndoGPT** - A GPT-based language model for Indonesian
   - Available on HuggingFace: [indobenchmark/indogpt](https://huggingface.co/indobenchmark/indogpt)

### Dataset
- **Indo4B-Plus Dataset** - A large pretraining dataset (~25 GB uncompressed)
  - Contains approximately 4 billion words of Indonesian text
  - Used for pretraining IndoBART and IndoGPT models

### Tokenization
The project implements custom tokenizers for Indonesian language processing:
- `IndoNLGTokenizer` - Custom tokenizer for IndoBART and IndoGPT
- `MBart52Tokenizer` - Extended MBart tokenizer

## Project Structure

### Core Modules
- **modules/** - Contains model-specific implementations
  - `tokenization_indonlg.py` - Implementation of the IndoNLG tokenizer
  - `tokenization_mbart52.py` - Implementation of the MBart52 tokenizer
  - `gpt2_custom_loss.py` - Custom loss functions for GPT-2 style models

### Training and Evaluation
- **main.py** - Main script for training models
- **evaluate.py** - Script for evaluating model performance

### Datasets
- **dataset/** - Contains dataset implementations and resources
  - `MT_JAVNRF_INZNTV/` - Javanese to Indonesian translation dataset
  - `MT_SUNIBS_INZNTV/` - Sundanese to Indonesian translation dataset

### Examples and Tutorials
- **examples/** - Example notebooks for using the models
  - `Finetuning_idsu_IndoBART.ipynb` - Example of fine-tuning IndoBART for Indonesian to Sundanese translation
  - `Finetuning_idsu_IndoBART-v2.ipynb` - Example of fine-tuning IndoBART-v2
  - `Finetuning_idsu_IndoGPT.ipynb` - Example of fine-tuning IndoGPT

### Utilities
- **utils/** - Various utility functions for data processing, training, and evaluation

## Features and Capabilities

The IndoNLG toolkit supports various NLP tasks:
1. **Machine Translation** - Between Indonesian and local languages (Javanese, Sundanese)
2. **Summarization** - Text summarization in Indonesian
3. **Text Generation** - Conditional and unconditional text generation

## Training & Fine-tuning

The repository includes code for:
- Pre-training IndoBART and IndoGPT on the Indo4B-Plus dataset
- Fine-tuning these models on specific downstream tasks
- Evaluation metrics for assessing model performance

## Usage Examples

The example notebooks demonstrate how to:
1. Load pre-trained models from HuggingFace
2. Fine-tune the models on specific tasks
3. Generate text using the fine-tuned models
4. Evaluate model performance using appropriate metrics

## Research Paper

IndoNLG has been published in EMNLP 2021. The paper can be found at: https://aclanthology.org/2021.emnlp-main.699

## Requirements

The project relies on PyTorch and the Transformers library, as specified in the requirements.txt file.

## License

The project is released under the MIT License.
