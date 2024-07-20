from src.imports.common_imports import *

"""
Positional encodings are vectors added to the input embeddings that provide information about the position of each 
token in the sequence. These vectors have the same dimension as the embeddings, allowing them to be summed directly.

The PositionalEncoding class is designed to generate and apply positional encodings to the input embeddings, allowing 
the model to capture the order of the tokens in a sequence.
"""

class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,       # Dimension of the embedding vectors
                 seq_len: int,       # Maximum sequence length
                 dropout_proba: float # Dropout probability
                 ) -> None:
        super().__init__()         # Initialize the parent class (nn.Module)
        self.d_model = d_model     # Save the dimension of the embedding
        self.seq_len = seq_len     # Save the sequence length
        self.dropout = nn.Dropout(dropout_proba) # Initialize the dropout layer

        # Create a matrix of shape (seq_len, d_model) to hold positional encodings
        pos_encoding = torch.zeros(seq_len, d_model)

        # Create a vector of shape (seq_len, 1) containing positions 0 to seq_len-1
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)

        # Calculate the div_term which is used to scale the positions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply the sine function to the even positions (0, 2, 4, ...)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)

        # Apply the cosine function to the odd positions (1, 3, 5, ...)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)

        # Add an extra dimension to match the input shape (1, seq_len, d_model)
        pos_encoding = pos_encoding.unsqueeze(0)

        # Register the positional encoding matrix as a buffer, so it's not considered a parameter - not updated during backprop
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        # Add the positional encodings to the input embeddings
        x = x + self.pos_encoding[:, :x.shape[1], :].detach()

        # Apply dropout to the result
        return self.dropout(x)

