from __future__ import unicode_literals, print_function, division
import torch
import math
import torch.nn as nn
from torch import LongTensor, FloatTensor
from torch.autograd import Variable
from RNN import GRU
import torch.nn.functional as F


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, cuda=False):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = GRU(hidden_size, hidden_size)
        self.cuda = cuda

    def forward(self, y, x, hidden):
        x = self.embedding(x).view(1, 1, -1)
        y = self.embedding(y).view(1, 1, -1)
        x, hidden = self.gru(y, x, hidden)
        return x[0], hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if self.cuda:
            return result.cuda()
        else:
            return result


class Attentive_Encoder(nn.Module):
    def __init__(self, vocab_size, d, q, cuda=False):
        super(Attentive_Encoder, self).__init__()
        self.hidden_size = d
        self.embedding = nn.Embedding(vocab_size, d)
        self.positions = nn.Embedding(100, d)
        self.q = q
        self.B = torch.FloatTensor(torch.rand((d, d, q)))
        self.B = nn.Parameter(self.B)
        self.cuda = cuda

    def forward(self, x, y, hidden):
        M = x.size()[0]
        x_pos = Variable(LongTensor([i for i in range(M)]))
        x_pos = self.positions(x_pos)
        x = self.embedding(x).squeeze(1)
        a = (x + x_pos).squeeze(1)

        half_q = int(math.floor(self.q/2))
        a = a.unsqueeze(0).transpose(1, 2)
        z = F.conv1d(a, self.B, padding=half_q).transpose(1, 2)

        # Very annoying feature: you cannot specify the axis for the softmax
        alphas = F.softmax((z @ hidden.transpose(1, 2)).transpose(0, 1))
        c = alphas.transpose(0, 2) @ x
        return c.transpose(0, 1)

    def initHidden(self):
        result = Variable(torch.rand((1, 1, self.hidden_size)))
        if self.cuda:
            return result.cuda()
        else:
            return result
