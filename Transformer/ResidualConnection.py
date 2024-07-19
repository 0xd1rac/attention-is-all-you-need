from imports import *
from ProjectionLayer import ProjectionLayer
from LayerNormalization import LayerNormalization

class ResidualConnection(nn.Module):
    def __init__(self, 
                dropout_proba: float
                ) -> None:
        """
        Initialize the ResidualConnection module.

        Args:
            dropout_proba (float): Dropout probability for regularization.
        """
        super().__init__()
        self.dropout = nn.Dropout(dropout_proba)  # Dropout layer for regularization
        self.norm = LayerNormalization()  # Layer normalization to stabilize training

    def forward(self, 
                x: torch.Tensor, 
                sublayer: Callable[[torch.Tensor], torch.Tensor]
                ) -> torch.Tensor:
        """
        Forward pass through the ResidualConnection module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            sublayer (Callable[[torch.Tensor], torch.Tensor]): A sublayer function or module
                (e.g., self-attention or feed-forward network) that will be applied to the input
                after normalization.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, seq_length, d_model) after applying
                the sublayer with residual connection, layer normalization, and dropout.
        """
        normalized_x = self.norm(x)  # Apply layer normalization to the input
        sublayer_output = sublayer(normalized_x)  # Apply the sublayer (e.g., self-attention or feed-forward)
        dropped_out_output = self.dropout(sublayer_output)  # Apply dropout to the sublayer output
        return x + dropped_out_output  # Add the original input (residual connection) to the transformed input
