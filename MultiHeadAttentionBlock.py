from imports import *

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, 
                d_model: int, 
                h: int, 
                dropout_proba: float
                ) -> None:
        """
        Initialize the MultiHeadAttentionBlock module.

        Args:
            d_model (int): Dimension of the model.
            h (int): Number of attention heads.
            dropout_proba (float): Dropout probability for regularization.
        """
        super().__init__()
        self.d_model = d_model
        self.h = h

        # Ensure d_model is divisible by h
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h  # Dimension of each attention head

        # Linear layers to project inputs to query, key, and value
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        # Linear layer to project concatenated outputs back to d_model
        self.W_O = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout_proba)

    @staticmethod
    def self_attention(q: torch.Tensor, 
                       k: torch.Tensor, 
                       v: torch.Tensor, 
                       dropout: nn.Dropout, 
                       attn_mask: torch.Tensor = None
                       ) -> torch.Tensor:
        """
        Compute self-attention.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, num_heads, seq_len, d_k).
            k (torch.Tensor): Key tensor of shape (batch_size, num_heads, seq_len, d_k).
            v (torch.Tensor): Value tensor of shape (batch_size, num_heads, seq_len, d_k).
            dropout (nn.Dropout): Dropout layer for regularization.
            attn_mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, 1, seq_len, seq_len). Default is None.

        Returns:
            torch.Tensor: Output tensor after applying self-attention and attention weights.
        """
        d_k = q.shape[-1]
        # Compute attention scores
        attention_scores = (q @ k.transpose(-2, -1)) / math.sqrt(d_k)
        
        # Apply attention mask if provided
        if attn_mask is not None:
            attention_scores.masked_fill(attn_mask == 0, float('-inf'))

        # Apply softmax to obtain attention weights
        attention_scores = F.softmax(attention_scores, dim=-1)

        # Apply dropout to attention weights
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        # Compute weighted sum of values
        return (attention_scores @ v), attention_scores

    def forward(self, 
                Q: torch.Tensor, 
                K: torch.Tensor, 
                V: torch.Tensor, 
                attn_mask: torch.Tensor = None
                ) -> torch.Tensor:
        """
        Forward pass through the MultiHeadAttentionBlock module.

        Args:
            Q (torch.Tensor): Query tensor of shape (batch_size, seq_len, d_model).
            K (torch.Tensor): Key tensor of shape (batch_size, seq_len, d_model).
            V (torch.Tensor): Value tensor of shape (batch_size, seq_len, d_model).
            attn_mask (torch.Tensor, optional): Attention mask tensor of shape (batch_size, 1, seq_len, seq_len). Default is None.

        Returns:
            torch.Tensor: Output tensor after applying multi-head self-attention and final linear transformation.
        """
        # Apply linear transformations to input Q, K, V
        query = self.W_Q(Q)
        key = self.W_K(K)
        value = self.W_V(V)

        # Split Q, K, V into multiple heads and rearrange dimensions
        # (batch, seq_len, d_model) -> (batch, seq_len, h, d_k) -> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

        # Perform self-attention
        x, self.attention_scores = MultiHeadAttentionBlock.self_attention(query, key, value, self.dropout, attn_mask)

        # Concatenate heads and reshape back to original dimensions
        # (batch, h, seq_len, d_k) -> (batch, seq_len, h, d_k) -> (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        # Final linear transformation
        # (batch, seq_len, d_model) -> (batch, seq_len, d_model)
        x = self.W_O(x)
        return x
