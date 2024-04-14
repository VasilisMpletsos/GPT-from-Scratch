import math

import numpy as np
import torch
import torch.nn as nn


# Creating the Positional Encoding
class PositionEncoder(nn.Module):
    def __init__(self, output_dimensions: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.output_dimensions = output_dimensions  # Dimensionality of the model
        self.seq_len = seq_len  # Maximum sequence length
        self.dropout = nn.Dropout(dropout)  # Dropout layer to prevent overfitting

        # Creating a positional encoding matrix of shape (seq_len, output_dimensions) filled with zeros
        pe = torch.zeros(seq_len, output_dimensions)

        # Creating a tensor representing positions (0 to seq_len - 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(
            1
        )  # Transforming 'position' into a 2D tensor['seq_len, 1']

        # Creating the division term for the positional encoding formula
        div_term = torch.exp(
            torch.arange(0, output_dimensions, 2).float()
            * (-math.log(10000.0) / output_dimensions)
        )

        # Apply sine to even indices in pe
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices in pe
        pe[:, 1::2] = torch.cos(position * div_term)

        # Adding an extra dimension at the beginning of pe matrix for batch handling
        pe = pe.unsqueeze(0)

        # Registering 'pe' as buffer. Buffer is a tensor not considered as a model parameter
        self.register_buffer("pe", pe)

    def forward(self, x):
        # Addind positional encoding to the input tensor X
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)  # Dropout for regularization


class InputEmbeddings(nn.Module):
    """
    This is the initial input layer for the text.
    It will create all the words into embeddings.

    For example if we have a sentence with 10 words
    then we will get a matrix of 10(words)*output_dimension(embeddings) vectors

    Inputs:
    * output_dimension: The embeddings size
    * vocab_size: Is the number of total unique words that we will support in our vocab
    """

    # create the init function
    def __init__(self, output_dimension: int, vocab_size: int):
        super().__init__()
        self.output_dimension = output_dimension
        self.vocab_size = vocab_size
        # Initialize the embedding matrix
        self.embedding_layer = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=output_dimension
        )

    # create the forward pass how will behave
    def forward(self, x):
        embedding_calculation = self.embedding_layer(x)
        normalization_factor = np.sqrt(self.output_dimension)
        return embedding_calculation + normalization_factor


class LayerNorm(nn.Module):
    def __init__(self, eps: float = 10e-6):
        super().__init__()

        # A small value to never divide by 0
        self.eps = eps

        # define a trainable parameter a and initialize with ones, same with bias
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.ones(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        standarized_output = (x - mean) / (std + self.eps)
        return self.alpha * standarized_output + self.bias


class FeedForward(nn.Module):
    def __init__(self, input_dim: int, between_dimension: int, dropout: float):
        super().__init__()

        self.linear1 = nn.Linear(input_dim, between_dimension)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(between_dimension, input_dim)

    def forward(self, x):
        pass_1 = torch.relu(self.linear1(x))
        pass_drop = self.dropout(pass_1)
        return self.linear2(pass_drop)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        # We ensure that the dimensions of the model is divisible by the number of heads
        assert d_model % num_heads == 0, "d_model is not divisible by num_heads"

        # get the dimension for each head key query and value
        self.d_k = d_model // num_heads

        # define the weight matrices
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        # and finally the dropout
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        query_last_dim = query.shape[-1]

        # calculate attention scores
        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(query_last_dim)

        # hide mask words
        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        # extract the probabilities
        attention_scores = attention_scores.softmax(dim=-1)

        # apply dropout
        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        # calculate query key values weights
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_k(v)

        # Split results into smaller matrices to be fed in heads
        query = query.view(
            query.shape[0], query.shape[1], self.num_heads, self.d_k
        ).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(
            1, 2
        )
        value = value.view(
            value.shape[0], value.shape[1], self.num_heads, self.d_k
        ).transpose(1, 2)

        # Get the outpout of the attention calculation
        x, self.attention_scores = MultiHeadAttention.attention(
            query, key, value, mask, self.dropout
        )

        # have the results in single rows
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(x.shape[0], -1, self.num_heads * self.d_k)
        )

        return self.w_o(x)


class Residual(nn.Module):
    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNorm()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class ProjectionLayer(nn.Module):
    def __init__(self, input_dim: int, voacb_size: int):
        super().__init__()
        self.projection = nn.Linear(input_dim, voacb_size)

    def forward(self, x):
        return torch.log_softmax(self.projection(x), dim=-1)
