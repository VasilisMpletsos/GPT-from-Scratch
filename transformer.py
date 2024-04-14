import torch.nn as nn

from blocks import (
    FeedForward,
    InputEmbeddings,
    MultiHeadAttention,
    PositionEncoder,
    ProjectionLayer,
)
from decoder import Decoder, DecoderBlock
from encoder import Encoder, EncoderBlock


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embeddings: InputEmbeddings,
        trgt_embeddings: InputEmbeddings,
        src_positional: PositionEncoder,
        trgt_positional: PositionEncoder,
        projection_layer: ProjectionLayer,
    ) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.src_embeddings = src_embeddings
        self.src_positional = src_positional
        self.trgt_embeddings = trgt_embeddings
        self.trgt_positional = trgt_positional
        self.projection_layer = projection_layer

    def encode(self, src, src_mask):
        src = self.src_embeddings(src)
        src = self.src_positional(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, target, target_mask):
        target = self.trgt_embeddings(target)
        target = self.trgt_positional(target)
        return self.decoder(target, encoder_output, src_mask, target_mask)

    def project(self, x):
        return self.projection_layer(x)


# Definin function and its parameter, including model dimension, number of encoder and decoder stacks, heads, etc.
def build_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    src_seq_len: int,
    tgt_seq_len: int,
    d_model: int = 512,
    N: int = 6,
    h: int = 8,
    dropout: float = 0.1,
    d_ff: int = 2048,
) -> Transformer:
    # Creating Embedding layers
    src_embed = InputEmbeddings(
        d_model, src_vocab_size
    )  # Source language (Source Vocabulary to 512-dimensional vectors)
    tgt_embed = InputEmbeddings(
        d_model, tgt_vocab_size
    )  # Target language (Target Vocabulary to 512-dimensional vectors)

    # Creating Positional Encoding layers
    src_pos = PositionEncoder(
        d_model, src_seq_len, dropout
    )  # Positional encoding for the source language embeddings
    tgt_pos = PositionEncoder(
        d_model, tgt_seq_len, dropout
    )  # Positional encoding for the target language embeddings

    # Creating EncoderBlocks
    encoder_blocks = []  # Initial list of empty EncoderBlocks
    for _ in range(N):  # Iterating 'N' times to create 'N' EncoderBlocks (N = 6)
        encoder_self_attention_block = MultiHeadAttention(
            d_model, h, dropout
        )  # Self-Attention
        feed_forward_block = FeedForward(d_model, d_ff, dropout)  # FeedForward

        # Combine layers into an EncoderBlock
        encoder_block = EncoderBlock(
            encoder_self_attention_block, feed_forward_block, dropout
        )
        encoder_blocks.append(
            encoder_block
        )  # Appending EncoderBlock to the list of EncoderBlocks

    # Creating DecoderBlocks
    decoder_blocks = []  # Initial list of empty DecoderBlocks
    for _ in range(N):  # Iterating 'N' times to create 'N' DecoderBlocks (N = 6)
        decoder_self_attention_block = MultiHeadAttention(
            d_model, h, dropout
        )  # Self-Attention
        decoder_cross_attention_block = MultiHeadAttention(
            d_model, h, dropout
        )  # Cross-Attention
        feed_forward_block = FeedForward(d_model, d_ff, dropout)  # FeedForward

        # Combining layers into a DecoderBlock
        decoder_block = DecoderBlock(
            decoder_self_attention_block,
            decoder_cross_attention_block,
            feed_forward_block,
            dropout,
        )
        decoder_blocks.append(
            decoder_block
        )  # Appending DecoderBlock to the list of DecoderBlocks

    # Creating the Encoder and Decoder by using the EncoderBlocks and DecoderBlocks lists
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    # Creating projection layer
    projection_layer = ProjectionLayer(
        d_model, tgt_vocab_size
    )  # Map the output of Decoder to the Target Vocabulary Space

    # Creating the transformer by combining everything above
    transformer = Transformer(
        encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer
    )

    # Initialize the parameters
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer
