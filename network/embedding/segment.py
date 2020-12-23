import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512, mini_batch_size=4):
        super().__init__(5, embed_size)
