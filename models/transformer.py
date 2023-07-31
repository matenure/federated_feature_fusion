import math
from typing import Tuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
# from torch.utils.data import dataset

class TransformerModel(nn.Module):

    def __init__(self, seq_len:int, d_input: int, d_model: int, nhead: int, d_hid: int,
                 nlayers: int, dropout: float = 0.5):
        super().__init__()
        self.model_type = 'Transformer'
        self.pos_encoder = PositionalEncoding(d_model, seq_len, dropout=0.1)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, d_hid, dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        # self.encoder = nn.Embedding(ntoken, d_model)
        self.encoder = nn.Linear(d_input, d_model,bias=False)
        self.d_model = d_model
        # self.decoder = nn.Linear(d_model, ntoken)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.pos_encoder.position_x.data.uniform_(0,180)


    def forward(self, src: Tensor, self_independent_pos = True) -> Tensor:
        """
        Args:
            src: Tensor, shape [batch_size, seq_len, d_input]

        Returns:
            output Tensor of shape [batch_size, seq_len, d_model]
        """
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src, self_independent_pos)
        output = self.transformer_encoder(src)
        return output


def generate_square_subsequent_mask(sz: int) -> Tensor:
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)



# class PositionalEncoding(nn.Module):
#
#     def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
#         super().__init__()
#         self.dropout = nn.Dropout(p=dropout)
#
#         position = torch.arange(max_len).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
#         pe = torch.zeros(max_len, 1, d_model)
#         pe[:, 0, 0::2] = torch.sin(position * div_term)
#         pe[:, 0, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe)
#
#     def forward(self, x: Tensor) -> Tensor:
#         """
#         Args:
#             x: Tensor, shape [seq_len, batch_size, embedding_dim]
#         """
#         x = x + self.pe[:x.size(0)]
#         return self.dropout(x)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, seq_len: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        # self.pe = nn.Parameter(torch.Tensor(max_len, 1, d_model))
        # self.pe.data.zeros_()
        self.seq_len = seq_len
        self.d_model = d_model
        self.position_x = nn.Parameter(torch.Tensor(seq_len,1))
        self.lin = nn.Linear(d_model, 1)
        # self.position_y = nn.Parameter(torch.Tensor(max_len,1))


    def forward(self, x: Tensor, self_independent = False) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))


        if self_independent:
            pe = torch.zeros_like(x)
            self.px = self.lin(x)
            pe[:, :, 0::2] = torch.sin(self.px * div_term)
            pe[:, :, 1::2] = torch.cos(self.px * div_term)
        else:
            pe = torch.zeros(1, self.seq_len, self.d_model)
            self.px = self.position_x
            pe[0, :, 0::2] = torch.sin(self.px * div_term)
            pe[0, :, 1::2] = torch.cos(self.px * div_term)
        x = x + pe

        return self.dropout(x)
