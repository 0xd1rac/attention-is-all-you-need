from imports import *
from Decoder import Decoder
from DecoderBlock import DecoderBlock
from Encoder import Encoder
from EncoderBlock import EncoderBlock
from FeedForwardBlock import FeedForwardBlock
from InputEmbeddings import InputEmbeddings
from LayerNormalization import LayerNormalization
from MultiHeadAttentionBlock import MultiHeadAttentionBlock
from PositionalEncoding import PositionalEncoding
from ProjectionLayer import ProjectionLayer
from ResidualConnection import ResidualConnection
from Transformer import Transformer

class TransformerBuilder:
    @staticmethod
    def build_transformer(src_vocab_size: int,
                          tgt_vocab_size: int,
                          src_seq_len: int,
                          tgt_seq_len: int,
                          d_model: int = 512,
                          N: int = 6,  # Number of encoder and decoder blocks
                          h: int = 8,  # Number of heads in MHA block
                          dropout_proba: float = 0.1,
                          d_ff: int = 2048  # Number of parameters in the Feed Forward Layer
                          ) -> Transformer:
        """
        Build a Transformer model.

        Args:
            src_vocab_size (int): Size of the source vocabulary.
            tgt_vocab_size (int): Size of the target vocabulary.
            src_seq_len (int): Length of the source sequence.
            tgt_seq_len (int): Length of the target sequence.
            d_model (int): Dimension of the model. Default is 512.
            N (int): Number of encoder and decoder blocks. Default is 6.
            h (int): Number of heads in the multi-head attention block. Default is 8.
            dropout_proba (float): Dropout probability for regularization. Default is 0.1.
            d_ff (int): Number of parameters in the feed-forward layer. Default is 2048.

        Returns:
            Transformer: The constructed Transformer model.
        """
        # Create the embedding layers
        src_embed = InputEmbeddings(d_model, src_vocab_size)
        tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

        # Create the positional encoding layers
        src_pos = PositionalEncoding(d_model, src_seq_len, dropout_proba)
        tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout_proba)

        # Create the encoder blocks
        encoder_blocks = []
        for _ in range(N):
            self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout_proba)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_proba)
            encoder_block = EncoderBlock(self_attention_block, feed_forward_block, dropout_proba)
            encoder_blocks.append(encoder_block)

        # Create the decoder blocks
        decoder_blocks = []
        for _ in range(N):
            self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout_proba)
            cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout_proba)
            feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout_proba)
            decoder_block = DecoderBlock(self_attention_block, cross_attention_block, feed_forward_block, dropout_proba)
            decoder_blocks.append(decoder_block)

        # Create encoder and decoder
        encoder = Encoder(nn.ModuleList(encoder_blocks))
        decoder = Decoder(nn.ModuleList(decoder_blocks))

        # Create the projection layer
        projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

        # Create the transformer
        transformer = Transformer(encoder=encoder,
                                  decoder=decoder,
                                  src_embed=src_embed,
                                  tgt_embed=tgt_embed,
                                  src_pos=src_pos,
                                  tgt_pos=tgt_pos,
                                  projection_layer=projection_layer)

        # Initialize the parameters
        for p in transformer.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        return transformer
