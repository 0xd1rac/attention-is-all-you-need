from src.imports.common_imports import *

class FeedForwardBlock(nn.Module):
    def __init__(self,
                 d_model: int,       # Dimension of the embedding vector
                 d_ff: int,          # Dimension of the feed-forward layer
                 dropout_proba: float # Dropout probability
                 ) -> None:
        
        super().__init__()         # Initialize the parent class (nn.Module)
        self.linear_1 = nn.Linear(d_model, d_ff) # First linear transformation (batch_size, seq_len,d_model) -> (batch_size, seq_len, d_ff)
        self.dropout = nn.Dropout(dropout_proba) # Dropout layer
        self.linear_2 = nn.Linear(d_ff, d_model) # Second linear transformation (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)

    def forward(self, x):
        x = self.linear_1(x)       # Apply the first linear transformation
        x = F.relu(x)              # Apply ReLU activation function
        x = self.dropout(x)        # Apply dropout for regularization
        x = self.linear_2(x)       # Apply the second linear transformation
        return x                   # Return the output
