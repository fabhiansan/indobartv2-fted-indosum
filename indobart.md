Comprehensive Research on IndoBART-v2
1. Architecture Foundations
IndoBART-v2 is an advanced version of the original IndoBART, built on the BART (Bidirectional and Auto-Regressive Transformers) architecture. Key modifications include:

Pre-training Objectives: Sentence shuffling and token masking[1][4]
Tokenizer: Extended to support Indonesian, Sundanese, and Javanese languages[2][10]
Base Configuration: Maintains the same architecture as IndoBERT:
12 hidden layers (768 dimensions each)
12 attention heads
3,072-dimensional feed-forward layers[5]
2. Training Data Composition
The model is trained on the Indo4B-Plus corpus, a 26GB dataset that includes:

Indonesian Wikipedia: 74 million words
News Articles: 55 million words from Kompas, Tempo, and Liputan6
Web-Crawled Text: 90 million words
Common Crawl Data: Additional web content
Regional Language Content: Texts in Sundanese and Javanese[4][7][10]
3. Pre-training Methodology
The training process involves:

Dynamic Masking: 15% of tokens are randomly masked in each sequence[1]
Sentence Permutation: Original sentence order is randomly shuffled[1]
Optimization: AdamW optimizer with a learning rate of 1e-4[1][4]
Batch Processing: 128 sequences of 512 tokens each[5]
Infrastructure: Trained on 4x NVIDIA V100 GPUs for approximately 2 months[5]
# Example training configuration from IndoNLG toolkit:
from transformers import BartConfig

config = BartConfig(
    vocab_size=32032,  # Extended from original 30,521
    max_position_embeddings=512,
    encoder_layers=12,
    decoder_layers=12,
    attention_heads=12
)
4. Vocabulary Extension Process
The tokenizer was expanded to include:

New Regional Tokens: 1,511 new tokens identified through corpus analysis[2]
Preservation of Original Tokenization: Maintains the original IndoBERT tokenization for Latin-alphabet text[2]
Initialization of New Embeddings: New embeddings are initialized as the mean of existing vectors[2]
Resulting Vocabulary Size: Increased from 30,521 to 32,032 tokens[2]
5. Performance Optimization
Key technical decisions for performance optimization include:

Early Stopping: Training is halted if validation loss plateaus for 5 epochs[1]
Mixed Precision Training: FP16 optimizations for memory efficiency[5]
Sequence Packing: Texts are concatenated and divided into fixed 128-token windows[2]
6. Applications and Evaluation
IndoBART-v2 has been applied to various tasks, demonstrating its versatility and effectiveness:

Recipe Generation: Achieved 79.6% accuracy in food instruction synthesis[1]
Machine Translation: Outperforms PBSMT systems in local language to Indonesian tasks[11]
Cultural Reasoning: Scores 82.2% on location-aware disambiguation tests[3]
Performance Comparison (BLEU scores on recipe generation):
| Model        | BLEU-1 | BLEU-4 |
|--------------|--------|--------|
| Vanilla BART | 0.312  | 0.207  |
| IndoBART-v2  | 0.447  | 0.381  | [1]

7. Collaborative Development
The development of IndoBART-v2 is a collaborative effort involving:

Academic Institutions: Institut Teknologi Bandung, Universitas Indonesia
Industry Partners: Gojek, Prosa.AI The focus is on preserving Indonesian linguistic nuances while maintaining compatibility with existing NLP frameworks[7][10].
8. Future Work
Future research aims to:

Explore larger parameter variants
Develop multilingual extensions
Address resource constraints in low-resource language support[1][2]
References
IndoBART Fine-tuning for Indonesian Recipe Generator
NusaBERT: Extending IndoBERT and IndoBART with Regional Language Tokens
IndoBART-v2: A State-of-the-Art Model for Indonesian NLP
IndoBART-v2 on Hugging Face
IndoNLG Toolkit
IndoCareer: A Comprehensive Study of Indonesian NLP Models
IndoNLP GitHub Repository
Indo4B-Plus Dataset
IndoBART-v2 Training Configuration
IndoBART-v2 Tokenizer Implementation
IndoBART-v2 in Machine Translation
This research provides a comprehensive overview of IndoBART-v2, highlighting its architecture, training data, pre-training methodology, tokenizer, performance optimization, applications, and collaborative development. The model's advancements and performance metrics demonstrate its significant contributions to Indonesian NLP.