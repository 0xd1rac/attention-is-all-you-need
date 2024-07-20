# Attention Is All You Need

## Introduction
This repository contains the implementation of the paper "Attention Is All You Need" by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Łukasz Kaiser, and Illia Polosukhin. The paper was presented at the 31st Annual Conference on Neural Information Processing Systems (NeurIPS 2017).

The paper introduces the Transformer model, a novel architecture for sequence modeling that relies entirely on self-attention mechanisms, eschewing the recurrent and convolutional layers traditionally used in such models. The Transformer model has since become a cornerstone in the field of natural language processing (NLP), powering state-of-the-art models like BERT, GPT, and T5.

Key contributions of the paper include:
- **Self-Attention Mechanism:** The core idea of the Transformer is the use of self-attention to process input sequences. This mechanism allows the model to weigh the importance of different words in a sequence when encoding a particular word.
- **Positional Encoding:** Since the model does not use recurrent structures, it incorporates positional encodings to maintain the order of the sequence.
- **Scalability and Parallelization:** The architecture is highly parallelizable, making it more efficient to train on large datasets compared to recurrent models.

This implementation aims to replicate the results presented in the paper and provide a platform for further experimentation and research.

## Implementation Details
- **Frameworks Used:** PyTorch
- **Dataset:** WMT 2014 English-to-German translation task
- **Model Architecture:** The model consists of an encoder and decoder, each made up of multiple layers of self-attention and feedforward neural networks.


## Usage
To run this code:
1. Clone the repository
   ```sh
   git clone https://github.com/0xd1rac/attention-is-all-you-need.git
