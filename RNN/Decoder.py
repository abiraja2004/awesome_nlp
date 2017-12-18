from __future__ import unicode_literals, print_function, division
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F


class Elman_Cell(nn.Module):
    """Hidden cell for Elman architecture."""
    def __init__(self, d, bias=True):
        super(Elman_Cell, self).__init__()
        self.d = d

        # Modules
        self.w1 = nn.Linear(d, d, bias=bias)
        self.w2 = nn.Linear(d, d, bias=bias)
        self.w3 = nn.Linear(d, d, bias=bias)

    def forward(self, y, c, hidden):
        return F.sigmoid(self.w1(y) + self.w2(c) + self.w3(hidden))


class Elman_Decoder(nn.Module):
    """Simple Elman decoder architecture for encoder-decoder
    seq-2-seq model."""
    def __init__(self, d, V, embed, max_length=10, enable_cuda=False):
        super(Elman_Decoder, self).__init__()
        self.d = d
        self.bias = True
        self._load_cell()
        self.d = d
        self.V = V
        self.max_length = max_length
        self.enable_cuda = enable_cuda
        self.w4 = nn.Linear(self.d, self.V)
        self.w5 = nn.Linear(self.d, self.V)

    def forward(self, y, hidden, c):
        # Calculate hidden state
        hidden = self.cell(y, c, hidden)

        # Calculate probs for every word in vocabulary
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

    def _load_cell(self):
        self.cell = Elman_Cell(self.d, self.bias)
