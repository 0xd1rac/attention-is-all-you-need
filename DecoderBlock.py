from imports import *
from MultiHeadAttentionBlock import MultiHeadAttentionBlock
from FeedForwardBlock import FeedForwardBlock
from ResidualConnection import ResidualConnection
from LayerNormalization import LayerNormalization

class DecoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 cross_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout_proba: float
                 ) -> None:
        """
        Initialize the DecoderBlock module.

        Args:
            self_attention_block (MultiHeadAttentionBlock): Instance of the multi-head self-attention block.
            cross_attention_block (MultiHeadAttentionBlock): Instance of the multi-head cross-attention block.
            feed_forward_block (FeedForwardBlock): Instance of the feed-forward block.
            dropout_proba (float): Dropout probability for regularization.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        
        # Initialize residual connections for self-attention, cross-attention, and feed-forward blocks
        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout_proba) for _ in range(3)
        ])

    def forward(self,
                x: torch.Tensor,
                encoder_output: torch.Tensor,
                encoder_output_mask: torch.Tensor,
                decoder_mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the DecoderBlock module.

        Args:
            x (torch.Tensor): Input tensor from the previous decoder layer (or input embedding), shape (batch_size, seq_length, d_model).
            encoder_output (torch.Tensor): Output tensor from the encoder, shape (batch_size, seq_length, d_model).
            encoder_output_mask (torch.Tensor): Mask tensor applied to the encoder output, shape (batch_size, 1, seq_length, seq_length).
            decoder_mask (torch.Tensor): Mask tensor applied to the decoder input, shape (batch_size, 1, seq_length, seq_length).

        Returns:
            torch.Tensor: Output tensor after applying self-attention, cross-attention, and feed-forward blocks with residual connections, shape (batch_size, seq_length, d_model).
        """
        # Apply self-attention block with residual connection
        # Lambda function is used to correctly pass the multiple arguments (query, key, value, decoder_mask) to the self-attention block
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, decoder_mask))
        
        # Apply cross-attention block with residual connection
        # Lambda function is used to correctly pass the multiple arguments (query, key, value, encoder_output_mask) to the cross-attention block
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, encoder_output_mask))
        
        # Apply feed-forward block with residual connection
        # Direct function call since feed-forward block only needs the input tensor
        x = self.residual_connections[2](x, self.feed_forward_block)
        
        return x
