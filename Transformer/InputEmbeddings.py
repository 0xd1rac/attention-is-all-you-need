from imports import *

"""
The InputEmbeddings class converts token indices into dense enbedding vectors.
"""

class InputEmbeddings(nn.Module):
    def __init__(self,
                 d_model: int,   # Dimension of the embedding vector
                 vocab_size: int # Size of the vocabulary
                 ):
        super().__init__()  # Initialize the parent class (nn.Module)
        self.d_model = d_model  # Save the dimension of the embedding
        self.vocab_size = vocab_size  # Save the vocabulary size
        self.embedding = nn.Embedding(vocab_size, d_model)  # Create an embedding layer with vocab_size entries, each of dimension d_model

    def forward(self, x):
        # Forward pass through the embedding layer
        # Multiply the output of the embedding layer by the square root of the embedding dimension (d_model)
        # This scaling is often used in transformer models to stabilize training
        return self.embedding(x) * math.sqrt(self.d_model)

