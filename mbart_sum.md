# mBART for Indonesian Summarization

## Overview of mBART
- **Definition**: mBART is a sequence-to-sequence denoising auto-encoder pre-trained on large-scale monolingual corpora in many languages using the BART objective.
- **Pre-training**: The model is trained by noising phrases and permuting sentences, and a single Transformer model is learned to recover the texts.
- **Applications**: mBART has been fine-tuned for summarization tasks in various languages, including Indonesian.

## mBART in Indonesian Context
- **IndoBART**: A BART-based sequence-to-sequence model specifically adapted for Indonesian, available on Hugging Face.
- **IndoNLG Toolkit**: Supports various NLP tasks, including summarization in Indonesian.
- **IndoSUM Dataset**: Used for fine-tuning IndoBART for Indonesian text summarization.

## Research Findings
- **Indonesian News Text Summarization Using MBART Algorithm**: This research explores the use of mBART for text summarization in Bahasa Indonesia, highlighting its novelty and contributions to understanding challenges and opportunities in summarization techniques for Indonesian.
- **Fine-tuning mBART for Monolingual Summarization**: Discussions on fine-tuning mBART for summarization in one of the languages used during pre-training, including Indonesian.
- **Bidirectional and Auto-Regressive Transformer (BART) for Indonesian Abstractive Text Summarization**: Research aims to produce abstractive summaries from Indonesian language texts using BART, which can be extended to mBART.

## Practical Applications
- **IndoBART-v2**: A state-of-the-art model for Indonesian NLP, fine-tuned for various tasks including summarization.
- **IndoNLG Toolkit**: Provides resources and code for fine-tuning models like IndoBART for specific tasks such as summarization.

## Conclusion
mBART has significant potential for summarization in the Indonesian language, as evidenced by both theoretical research and practical applications. The IndoBART model, part of the IndoNLG toolkit, is specifically fine-tuned for Indonesian and supports summarization tasks. Further research and fine-tuning can enhance its performance and applicability in various Indonesian NLP tasks.

## Recommendations
- **Fine-tuning**: Continue fine-tuning IndoBART on specific datasets like IndoSUM for improved summarization performance.
- **Evaluation**: Use benchmark datasets and metrics to evaluate the summarization capabilities of mBART in Indonesian.
- **Collaboration**: Engage with the IndoNLG community and other researchers to share findings and improve models.