from src.imports.common_imports import *
from .LayerNormalization import LayerNormalization

class Encoder(nn.Module):
    def __init__(self, 
                encoder_blocks: nn.ModuleList
                ) -> None:
        """
        Initialize the Encoder module.

        Args:
            encoder_blocks (nn.ModuleList): List of encoder blocks to be applied sequentially.
        """
        super().__init__()
        self.encoder_blocks = encoder_blocks  # Store the list of encoder blocks
        self.norm = LayerNormalization()  # Initialize layer normalization

    def forward(self, 
                x: torch.Tensor, 
                mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the Encoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            mask (torch.Tensor): Mask tensor for attention mechanism, used to mask certain positions.

        Returns:
            torch.Tensor: Output tensor after applying all encoder blocks and final normalization.
        """
        # Pass the input tensor through each encoder block sequentially
        for block in self.encoder_blocks:
            x = block(x, mask)
        
        # Apply layer normalization to the final output
        return self.norm(x)
