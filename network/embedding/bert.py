import torch.nn as nn
import torch
from .position import PositionalEmbedding
from .segment import SegmentEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """

    def __init__(self, embed_size, dropout=0.1, mini_batch_size=2, sample_num=20):
        """
        :param vocab_size: total vocab size
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        self.position = PositionalEmbedding(d_model=embed_size, mini_batch_size=mini_batch_size, sample_num=sample_num)
        self.segment = SegmentEmbedding(embed_size=embed_size, mini_batch_size=mini_batch_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size
        self.sample_num = sample_num
        self.mini_batch_size = mini_batch_size

    def forward(self, sequence, segment_label):
        seg_embedding = self.segment(segment_label)
        pos_embedding = self.position()
        x = sequence + pos_embedding + seg_embedding
        return self.dropout(x)
