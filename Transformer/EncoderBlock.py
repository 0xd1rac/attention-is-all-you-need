from imports import *
from MultiHeadAttentionBlock import MultiHeadAttentionBlock
from FeedForwardBlock import FeedForwardBlock
from ResidualConnection import ResidualConnection

class EncoderBlock(nn.Module):
    def __init__(self,
                 self_attention_block: MultiHeadAttentionBlock,
                 feed_forward_block: FeedForwardBlock,
                 dropout_proba: float
                 ) -> None:
        """
        Initialize the EncoderBlock module.

        Args:
            self_attention_block (MultiHeadAttentionBlock): Instance of the multi-head self-attention block.
            feed_forward_block (FeedForwardBlock): Instance of the feed-forward block.
            dropout_proba (float): Dropout probability for regularization.
        """
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block

        # Initialize residual connections for both the self-attention and feed-forward blocks
        self.residual_connections = nn.ModuleList([
            ResidualConnection(dropout_proba) for _ in range(2)
        ])

    def forward(self, 
                x: torch.Tensor, 
                src_mask: torch.Tensor
                ) -> torch.Tensor:
        """
        Forward pass through the EncoderBlock module.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_length, d_model).
            src_mask (torch.Tensor): Source mask tensor for attention mechanism, used to mask certain positions.

        Returns:
            torch.Tensor: Output tensor after applying the self-attention and feed-forward blocks with residual connections.
        """
        # Apply self-attention block with residual connection
        
        # The lambda function is used for the self-attention block to handle the multiple arguments required 
        # (query, key, value, and source mask). The feed-forward block, needing only the input tensor, can be passed 
        # directly without a lambda function. 
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        
        # Apply feed-forward block with residual connection
        x = self.residual_connections[1](x, self.feed_forward_block)
        
        return x
