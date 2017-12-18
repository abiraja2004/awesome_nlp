from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class RAN_Cell(nn.Module):

    def __init__(self, d, batch_size, bias=True):
        super(RAN_Cell, self).__init__()
        self.d = d

        # Modules
        self.w1 = nn.Linear(d, d, bias=bias)
        self.w2 = nn.Linear(d, d, bias=bias)
        self.w3 = nn.Linear(d, d, bias=bias)
        self.w4 = nn.Linear(d, d, bias=bias)
        self.w5 = nn.Linear(d, d, bias=bias)
        self.w6 = nn.Linear(d, d, bias=bias)
        self.__init_m(batch_size)

    def forward(self, y, c, hidden):
        i = F.sigmoid(self.w1(y) + self.w2(c) + self.w3(hidden))
        f = F.sigmoid(self.w4(y) + self.w5(c) + self.w6(hidden))
        m = self.former_m[-1] * f + i * c
        self.former_m.append(m)
        return m

    def __init_m(self, batch_size):
        self.former_m = [Variable(torch.FloatTensor(torch.zeros((1, self.d))))]


class RAN_Decoder(nn.Module):
    def __init__(self, d, V, embed, max_length, batch_size, enable_cuda=False):
        super(RAN_Decoder, self).__init__()
        self.d = d
        self.bias = True
        self._load_cell(batch_size)
        self.d = d
        self.V = V
        self.max_length = max_length
        self.enable_cuda = enable_cuda
        self.w4 = nn.Linear(self.d, self.V)
        self.w5 = nn.Linear(self.d, self.V)

    def forward(self, y, hidden, c):
        hidden = self.cell(y, c, hidden)
        output = F.log_softmax((self.w4(hidden) + self.w5(c)).transpose(0, 2)).transpose(0, 2)
        return output.squeeze(1), hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.d))
        if self.enable_cuda:
            return result.cuda()
        else:
            return result

    def _load_cell(self, batch_size):
        self.cell = RAN_Cell(self.d, batch_size, self.bias)


# class LSTM_Cell(nn.Module):

#     def __init__(self, d, bias=True, batch_size=1):
#         super(LSTM_Cell, self).__init__()
#         self.d = d

#         # Modules
#         self.w1 = nn.Linear(d, d, bias=bias)
#         self.w2 = nn.Linear(d, d, bias=bias)
#         self.w3 = nn.Linear(d, d, bias=bias)
#         self.w4 = nn.Linear(d, d, bias=bias)
#         self.w5 = nn.Linear(d, d, bias=bias)
#         self.w6 = nn.Linear(d, d, bias=bias)
#         self.__init_m(batch_size)

#     def __init_m(self, batch_size):
#         self.former_m = [Variable(torch.FloatTensor(torch.zeros((1, self.d))))]

#     def forward(self, y, c, hidden):
#         i = F.sigmoid(self.w1(y) + self.w2(c) + self.w3(hidden))
#         f = F.sigmoid(self.w4(y) + self.w5(c) + self.w6(hidden))
#         m = self.former_m[-1] * f + i * c
#         self.former_m.append(m)
#         return m


# class LSTM_Decoder(nn.Module):
#     def __init__(self, d, V,  dropout, embed, max_length=10, cuda=False):
#         super(LSTM_Decoder, self).__init__()
#         self.d = d
#         self.V = V
#         self.max_length = max_length
#         self.cuda

#         self.embeddings = nn.Embedding(self.V, self.d)
#         # Use pretrained weights, a numpy matrix of shape vocab_dim x embed_dim
#         if embed is not None:
#             self.embeddings.weight.data.copy_(torch.from_numpy(embed))
#         self.dropout = nn.Dropout(dropout)
#         self.w4 = nn.Linear(self.d, self.V)
#         self.w5 = nn.Linear(self.d, self.V)

#         self.d = d
#         self.bias = True
#         self._load_cell()

#     def _load_cell(self):
#         self.cell = LSTM_Cell(self.d, self.bias)

#     def forward(self, y, hidden, c):
#         y = self.embedding(y).view(1, 1, -1)
#         y = self.dropout(y)
#         batch_size = y.size(0)
#         seq_length = y.size(1)

#         output = []
#         for i in range(min(seq_length, self.max_length)):
#             hidden = self.cell(y[:, i, :], c, hidden=hidden)
#             output.append(hidden)
#         output = torch.cat(output, 1)
#         output = F.log_softmax(self.w4(output[0]))
#         return output, hidden

#     def initHidden(self):
#         result = Variable(torch.zeros(1, 1, self.d))
#         if self.cuda:
#             return result.cuda()
#         else:
#             return result