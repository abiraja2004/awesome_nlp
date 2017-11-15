import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn._functions.rnn import Recurrent, StackedRNN


class RAN(nn.Module):
    """
    Implementation of a Recurrent Additive Network

    Source:
        https://arxiv.org/pdf/1705.07393.pdf
    """
    def __init__(self, ninp, nhid, nlayers, dropout=0.5):
        super(RAN, self).__init__()
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout

        # 5 weight matrices:
        self.W_cx = nn.Parameter(torch.Tensor(nhid, ninp))  # content layer
        self.W_ih = nn.Parameter(torch.Tensor(nhid, nhid))  # input gate on output
        self.W_ix = nn.Parameter(torch.Tensor(nhid, ninp))  # input gate on input
        self.W_fh = nn.Parameter(torch.Tensor(nhid, nhid))  # forget gate on output
        self.W_fx = nn.Parameter(torch.Tensor(nhid, ninp))  # forget gate on input
        # 5 biases
        self.b_i = nn.Parameter(torch.Tensor(nhid))
        self.b_f = nn.Parameter(torch.Tensor(nhid))

        self.weights = [self.W_cx, self.W_ih, self.W_ix, self.W_fh, self.W_fx]
        self.biases = [self.b_i, self.b_f]
        self.weight_init()

    def weight_init(self):
        for weight in self.weights:
            nn.init.uniform(weight)
        for bias in self.biases:
            nn.init.constant(bias, 0)

    def forward(self, x, hidden):
        layer = (Recurrent(RANCell), )
        func = StackedRNN(layer, self.nlayers, dropout=self.dropout)
        nexth, output = func(x, hidden, ((self.weights, self.biases), ))
        return output, nexth


def RANCell(x, hidden, weights, biases):
    W_cx, W_ih, W_ix, W_fh, W_fx = weights
    b_i, b_f = biases

    content_t = F.linear(x, W_cx)
    i_t = F.sigmoid(F.linear(hidden, W_ih) + F.linear(x, W_ix, b_i))
    f_t = F.sigmoid(F.linear(hidden, W_fh) + F.linear(x, W_fx, b_f))
    c_t = i_t * content_t + f_t * hidden
    h_t = F.tanh(c_t)
    return h_t
