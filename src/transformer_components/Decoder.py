from src.imports.common_imports import *
from .LayerNormalization import LayerNormalization

class Decoder(nn.Module):
    def __init__(self,
                decoder_blocks: nn.ModuleList
                ) -> None:
        """
        Initialize the Decoder module.

        Args:
            decoder_blocks (nn.ModuleList): List of decoder blocks to be applied sequentially.
        """
        super().__init__()
        self.decoder_blocks = decoder_blocks  # Store the list of decoder blocks
        self.norm = LayerNormalization()  # Initialize layer normalization

    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                encoder_output_mask: torch.Tensor,
                decoder_mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the Decoder module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            encoder_output (torch.Tensor): Output tensor from the encoder, shape (batch_size, seq_length, d_model).
            encoder_output_mask (torch.Tensor): Mask tensor applied to the encoder output, shape (batch_size, 1, seq_length, seq_length).
            decoder_mask (torch.Tensor): Mask tensor applied to the decoder input, shape (batch_size, 1, seq_length, seq_length).

        Returns:
            torch.Tensor: Output tensor after applying all decoder blocks and final normalization.
        """
        # Pass the input tensor through each decoder block sequentially
        for block in self.decoder_blocks:
            x = block(x, encoder_output, encoder_output_mask, decoder_mask)
        
        # Apply layer normalization to the final output
        return self.norm(x)
