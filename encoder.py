import torch.nn as nn

from blocks import FeedForward, LayerNorm, MultiHeadAttention, Residual


class EncoderBlock(nn.Module):
    def __init__(
        self,
        attention_block: MultiHeadAttention,
        feed_forward_block: FeedForward,
        dropout: float,
    ) -> None:
        super().__init__()

        self.attention_block = attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([Residual(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # first residual
        x = self.residual_connections[0](
            x, lambda x: self.attention_block(x, x, x, src_mask)
        )
        # second residual
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
