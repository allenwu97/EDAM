import torch.nn as nn
import torch
import math


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model, max_len= 46 * 46, mini_batch_size=2, sample_num=20):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False
        self.d_model = d_model
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe)

        self.mini_batch_size = mini_batch_size
        self.sample_num = sample_num




    def forward(self):
        output = [self.pe for i in range(self.mini_batch_size * self.sample_num)]
        output = torch.stack(output, 0).view(1, -1, self.d_model)
        return output