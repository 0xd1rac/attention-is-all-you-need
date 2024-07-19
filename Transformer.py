from imports import *
from Encoder import Encoder
from Decoder import Decoder
from InputEmbeddings import InputEmbeddings
from PositionalEncoding import PositionalEncoding
from ProjectionLayer import ProjectionLayer

class Transformer(nn.Module):
    def __init__(self, 
                 encoder: Encoder, 
                 decoder: Decoder, 
                 src_embed: InputEmbeddings, 
                 tgt_embed: InputEmbeddings, 
                 src_pos: PositionalEncoding, 
                 tgt_pos: PositionalEncoding, 
                 projection_layer: ProjectionLayer) -> None:
        """
        Initialize the Transformer module.

        Args:
            encoder (Encoder): Encoder module of the transformer.
            decoder (Decoder): Decoder module of the transformer.
            src_embed (InputEmbeddings): Source input embeddings.
            tgt_embed (InputEmbeddings): Target input embeddings.
            src_pos (PositionalEncoding): Positional encoding for source inputs.
            tgt_pos (PositionalEncoding): Positional encoding for target inputs.
            projection_layer (ProjectionLayer): Linear projection layer to generate final output probabilities.
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer

    def encode(self, 
               src: torch.Tensor, 
               src_mask: torch.Tensor
               ) -> torch.Tensor:
        """
        Encode the source sequence.

        Args:
            src (torch.Tensor): Source input tensor of shape (batch_size, seq_length).
            src_mask (torch.Tensor): Source mask tensor of shape (batch_size, 1, seq_length, seq_length).

        Returns:
            torch.Tensor: Encoded representation of shape (batch_size, seq_length, d_model).
        """
        # Apply source embeddings and positional encoding
        x = self.src_embed(src)
        x = self.src_pos(x)
        
        # Pass through the encoder
        x = self.encoder(x, src_mask)
        return x

    def decode(self, 
               encoder_output: torch.Tensor, 
               src_mask: torch.Tensor, 
               tgt: torch.Tensor, 
               tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Decode the target sequence using the encoder output.

        Args:
            encoder_output (torch.Tensor): Encoded source representation of shape (batch_size, seq_length, d_model).
            src_mask (torch.Tensor): Source mask tensor of shape (batch_size, 1, seq_length, seq_length).
            tgt (torch.Tensor): Target input tensor of shape (batch_size, seq_length).
            tgt_mask (torch.Tensor): Target mask tensor of shape (batch_size, 1, seq_length, seq_length).

        Returns:
            torch.Tensor: Decoded representation of shape (batch_size, seq_length, d_model).
        """
        # Apply target embeddings and positional encoding
        x = self.tgt_embed(tgt)
        x = self.tgt_pos(x)
        
        # Pass through the decoder
        x = self.decoder(x, encoder_output, src_mask, tgt_mask)
        return x

    def project(self, 
                x: torch.Tensor
                ) -> torch.Tensor:
        """
        Project the decoder output to the vocabulary size.

        Args:
            x (torch.Tensor): Decoder output tensor of shape (batch_size, seq_length, d_model).

        Returns:
            torch.Tensor: Log probabilities of shape (batch_size, seq_length, vocab_size).
        """
        return self.projection_layer(x)
