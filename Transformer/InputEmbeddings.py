from imports import *

class InputEmbeddings(nn.Module):
    """
    The InputEmbeddings class converts token indices into dense embedding vectors.
    """
    def __init__(self, 
                d_model: int, 
                vocab_size: int
                ) -> None:
        """
        Initialize the InputEmbeddings module.

        Args:
            d_model (int): Dimension of the embedding vector.
            vocab_size (int): Size of the vocabulary.
        """
        super().__init__()  # Initialize the parent class (nn.Module)
        self.d_model = d_model  # Save the dimension of the embedding
        self.vocab_size = vocab_size  # Save the vocabulary size
        self.embedding = nn.Embedding(vocab_size, d_model)  # Create an embedding layer with vocab_size entries, each of dimension d_model

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the InputEmbeddings module.

        Args:
            x (torch.Tensor): Input tensor of token indices with shape (batch_size, seq_length).

        Returns:
            torch.Tensor: Output tensor of dense embedding vectors with shape (batch_size, seq_length, d_model).
        """
        # Forward pass through the embedding layer
        embeddings = self.embedding(x)
        
        # Multiply the output of the embedding layer by the square root of the embedding dimension (d_model)
        # This scaling is often used in transformer models to stabilize training
        scaled_embeddings = embeddings * math.sqrt(self.d_model)
        
        return scaled_embeddings
