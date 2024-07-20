from src.imports.common_imports import *

class ProjectionLayer(nn.Module):
    def __init__(self, 
                d_model: int, 
                vocab_size: int
                ) -> None:
        """
        Initialize the ProjectionLayer module.

        Args:
            d_model (int): Dimension of the model.
            vocab_size (int): Size of the vocabulary.
        """
        super().__init__()
        self.projection = nn.Linear(d_model, vocab_size)  # Linear layer to project from d_model to vocab_size

    def forward(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the ProjectionLayer module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, vocab_size) with log probabilities.
        """
        # Apply the linear projection
        # (batch_size, seq_length, d_model) -> (batch_size, seq_length, vocab_size)
        projected = self.projection(x)
        
        # Apply log softmax to obtain log probabilities
        # Log softmax is applied along the last dimension (vocab_size)
        log_probs = F.log_softmax(projected, dim=-1)
        
        return log_probs
