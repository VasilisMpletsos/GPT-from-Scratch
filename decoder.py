import torch.nn as nn

from blocks import FeedForward, LayerNorm, MultiHeadAttention, Residual


class DecoderBlock(nn.Module):
    def __init__(
        self,
        multi_head_attention: MultiHeadAttention,
        cross_attention: MultiHeadAttention,
        feed_forward: FeedForward,
        dropout: float,
    ) -> None:
        super().__init__()

        self.multi_head_attention = multi_head_attention
        self.cross_attention = cross_attention
        self.feed_forward = feed_forward
        self.residual_connections = nn.ModuleList(Residual(dropout) for _ in range(3))

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        # self attention block
        x = self.residual_connections[0](
            x, lambda x: self.multi_head_attention(x, x, x, tgt_mask)
        )
        # cross attention block
        x = self.residual_connections[1](
            x,
            lambda x: self.cross_attention(x, encoder_output, encoder_output, src_mask),
        )
        # feed forward
        x = self.residual_connections[2](x, self.feed_forward)

        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()

        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)
