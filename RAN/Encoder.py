from __future__ import unicode_literals, print_function, division
import torch
import math
import torch.nn as nn
from torch import LongTensor, FloatTensor
from torch.autograd import Variable
import torch.nn.functional as F


class Attentive_Encoder(nn.Module):
    """Attentive encoder for the recurrent architectures."""
    def __init__(self, vocab_size, d, q, embed, enable_cuda=False):
        super(Attentive_Encoder, self).__init__()
        self.hidden_size = d
        if enable_cuda:
            self.embeddings = nn.Embedding(vocab_size, d).cuda()
            self.drop = nn.Dropout(p=0.2, inplace=False)
            self.positions = nn.Embedding(100, d).cuda()
            self.B = nn.Parameter(FloatTensor(torch.rand((d, d, q))),
                                  requires_grad=True).cuda()
        else:
            self.embeddings = nn.Embedding(vocab_size, d)
            self.drop = nn.Dropout(p=0.2, inplace=False)
            self.positions = nn.Embedding(100, d)
            self.B = nn.Parameter(FloatTensor(torch.rand((d, d, q))),
                                  requires_grad=True)

        # Use pretrained weights, a numpy matrix of shape vocab_dim x embed_dim
        if embed is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(embed))
        self.q = q
        self.enable_cuda = enable_cuda

    def forward(self, x, hidden, y):
        y = self.embeddings(y)
        if len(y.size()) < 3:
            y = y.unsqueeze(1)
        y = self.drop(y)
        batch_size, M = x.size()

        # Create the embeddings of x positions
        if self.enable_cuda:
            x_pos = Variable(LongTensor([[i for i in range(M)]
                                         for j in range(batch_size)])).cuda()
        else:
            x_pos = Variable(LongTensor([[i for i in range(M)]
                                         for j in range(batch_size)]))
        x_pos = self.positions(x_pos)
        x = self.embeddings(x)
        x = self.drop(x)
        a = (x + x_pos).transpose(1, 2)
        z = F.conv1d(a, self.B, padding=int(math.floor(self.q/2)))

        # Very annoying feature: you cannot specify the axis for the
        alphas = F.softmax((torch.matmul(hidden, z)).transpose(0, 2))
        c = torch.matmul(alphas.transpose(0, 2), x)
        return c, y

    def initHidden(self, batch_size):
        result = Variable(torch.rand((batch_size, 1, self.hidden_size)))
        if self.enable_cuda:
            return result.cuda()
        else:
            return result
