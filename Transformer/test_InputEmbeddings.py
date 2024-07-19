from InputEmbeddings import InputEmbeddings
from imports import *

def test_input_embeddings():
    # Define the vocabulary
    vocab = {
        "hello": 0, 
        "world": 1, 
        "attention": 2, 
        "is": 3, 
        "all": 4, 
        "you": 5, 
        "need": 6
        }

    vocab_size = len(vocab)  # Size of the vocabulary
    d_model = 512  # Dimension of the embedding vectors

    # Initialize the InputEmbeddings class
    input_embeddings = InputEmbeddings(d_model, vocab_size)

    # Convert sentence to token indices using the vocabulary
    sentence = "hello all you need is attention"
    print(f"Sentence: {sentence}\n")
    tokens = sentence.split(" ")
    print(f"Tokens: {tokens}\n")

    token_indices = torch.tensor([vocab[word] for word in tokens])  # Convert words to indices
    print(f"Token indices: {token_indices}\n")

    # Pass token indices through the embedding layer
    embedded_vectors = input_embeddings(token_indices)

    # Print the output dense vectors
    print(f"Embedded Vectors:\n{embedded_vectors}\n")

    print(f"Embedding matrix shape:\n{embedded_vectors.shape}\n")

if __name__ == "__main__":
    test_input_embeddings()
