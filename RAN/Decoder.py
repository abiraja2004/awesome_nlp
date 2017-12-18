from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class RAN_Cell(nn.Module):
    """Memory cell for the RAN decoder architecture."""
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
        self.__init_m()

    def forward(self, y, c, hidden):
        # input gate
        i = F.sigmoid(self.w1(y) + self.w2(c) + self.w3(hidden))

        # forget gate
        f = F.sigmoid(self.w4(y) + self.w5(c) + self.w6(hidden))
        m = self.former_m[-1] * f + i * c
        self.former_m.append(m)
        return m

    def __init_m(self):
        self.former_m = [Variable(torch.FloatTensor(torch.zeros((1, self.d))))]

    def clear_m(self):
        self.__init_m()


class RAN_Decoder(nn.Module):
    """RAN Decoder for the abstractive summarization encoder-decoder
    architecture."""
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
        # Get hidden cell
        hidden = self.cell(y, c, hidden)

        # Apply softmax to compress probabilities for the next word
        output = F.log_softmax(
            (self.w4(hidden) + self.w5(c)).transpose(0, 2)
        ).transpose(0, 2)
        return output.squeeze(1), hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.d))
        if self.enable_cuda:
            return result.cuda()
        else:
            return result

    def _load_cell(self, batch_size):
        self.cell = RAN_Cell(self.d, batch_size, self.bias)
