from imports import *
from PositionalEncoding import PositionalEncoding

def test_positional_encoding():
    # Define parameters
    d_model = 512  # Dimension of the embedding vectors
    seq_len = 5   # Maximum sequence length
    dropout_proba = 0.1  # Dropout probability

    # Initialize the PositionalEncoding class
    positional_encoding = PositionalEncoding(d_model, seq_len, dropout_proba)

    # Create a sample input tensor of shape (batch_size, seq_len, d_model)
    batch_size = 2
    sample_input = torch.zeros((batch_size, seq_len, d_model))

    # Pass the sample input through the positional encoding layer
    output = positional_encoding(sample_input)

    # Print the output
    print(f"Positional Encodings Shape: {output.shape}\n")

    print(f"Positional Encodings added to the input tensor: {output}")
   



# Run the test function
if __name__ == "__main__":
    test_positional_encoding()