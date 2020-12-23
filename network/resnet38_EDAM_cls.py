import torch
import torch.nn as nn
import torch.nn.functional as F
from network.transformer import TransformerBlock
from network.embedding import BERTEmbedding
import network.resnet38d


class Net(network.resnet38d.Net):
    def __init__(self):
        super().__init__()
        self.mask_layer = nn.Conv2d(4096, 21, 3, padding=1, bias=False)
        torch.nn.init.xavier_uniform_(self.mask_layer.weight)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.dropout7 = torch.nn.Dropout2d(0.8)

        self.sample_num = 20

        self.hidden = 128
        self.mini_batch_size = 2
        self.attn_heads = 2
        self.n_layers = 1
        self.embedding = BERTEmbedding(embed_size=self.hidden, mini_batch_size=self.mini_batch_size,sample_num=self.sample_num)
        self.trans_list = nn.ModuleList([nn.ModuleList(
            [TransformerBlock(self.hidden, self.attn_heads, self.hidden * 4, 0.1) for _ in range(self.n_layers)]) for _ in range(20)])

        self.d_d = nn.Conv2d(4096, self.hidden, 1, bias=False)
        torch.nn.init.xavier_uniform_(self.d_d.weight)

        self.not_training = [self.conv1a, self.b2, self.b2_1, self.b2_2]
        self.from_scratch_layers = [self.mask_layer, self.d_d]

        self.fc_list = nn.ModuleList([nn.Conv2d(self.hidden, 1, 1, bias=False) for _ in range(20)])

        for fc in self.fc_list:
            torch.nn.init.xavier_uniform_(fc.weight)
            self.from_scratch_layers.append(fc)

        for transformer in self.trans_list:
            for p in transformer.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)

        for p in self.embedding.parameters():
            torch.nn.init.xavier_uniform_(p)



    def forward(self, img, label_idx):
        c, h, w = img.size()[-3:]
        x = img.view(-1, c, h, w)
        x = super().forward(x)

        n, c, h, w = x.size()
        assert n == self.mini_batch_size
        assert len(label_idx) == self.sample_num

        mask = self.mask_layer(x)
        mask = F.normalize(mask)
        mask = torch.abs(mask)
        label_feature = []
        x = self.d_d(x)
        x = x.permute(1, 0, 2, 3).contiguous()         #self.hidden,n,h,w
        mask = mask.permute(1, 0, 2, 3).contiguous()   #21,n,h,w
        for i in label_idx:
            feature = torch.mul(x, mask[i]).permute(1, 0, 2, 3) #n,c,h,w
            label_feature.append(feature)
        x = torch.stack(label_feature, 1).view(-1, self.hidden, h, w)   #n*self.sample_num,self.hidden,h,w

        x = x.view(n, self.sample_num, self.hidden, h, w).permute(0,1,3,4,2).contiguous()
        x = x.view(1, self.mini_batch_size*self.sample_num*h*w, self.hidden)

        segment_info = torch.zeros(1, self.mini_batch_size * self.sample_num * h * w)
        for i in range(self.mini_batch_size):
            segment_info[:, i * self.sample_num * h * w:(i + 1) * self.sample_num * h * w] = i
        segment_info = segment_info.to(torch.int64).cuda()
        x = self.embedding(x, segment_info)
        output = []
        for i in range(self.sample_num):
            trans_input = []
            for j in range(self.mini_batch_size):
                trans_input.append(x[:, i*h*w+j*self.sample_num*h*w:(i+1)*h*w+j*self.sample_num*h*w, :])
            trans_input = torch.stack(trans_input, dim=1).view(1, -1, self.hidden)     #1,self.mini_batch*h*w,self.hidden
            for block in self.trans_list[label_idx[i]]:
                trans_input = block.forward(trans_input, None)
            trans_output = trans_input.view(self.mini_batch_size, h, w, self.hidden).permute(0,3,1,2)
            trans_output = self.gap(trans_output)
            trans_output = self.fc_list[label_idx[i]](trans_output).view(-1)
            output.append(trans_output)
        x = torch.stack(output, dim=1).view(-1)  #self.mini_batch_size*self.sample_num

        return x

    def forward_cam(self, x):
        x = super().forward(x)
        mask = self.mask_layer(x)
        mask = F.normalize(mask)
        mask = torch.abs(mask)
        return mask

    def get_parameter_groups(self):
        groups = ([], [], [], [], [], [])
        for name,m in self.embedding.named_parameters():
            if m.requires_grad:
                groups[4].append(m)
        for transformer in self.trans_list:
            for name, m in transformer.named_parameters():
                if m.requires_grad:
                    groups[4].append(m)
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):

                if m.weight.requires_grad:
                    if m in self.from_scratch_layers:
                        groups[2].append(m.weight)
                    else:
                        groups[0].append(m.weight)

                if m.bias is not None and m.bias.requires_grad:

                    if m in self.from_scratch_layers:
                        groups[3].append(m.bias)
                    else:
                        groups[1].append(m.bias)

        for i in range(len(groups)):
            print(len(groups[i]))

        return groups
