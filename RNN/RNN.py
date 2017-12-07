import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class GRUCell(nn.Module):

    def __init__(self, input_size, hidden_size,
                       bias_ih=True, bias_hh=False):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh

        # Modules
        self.w1 = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.w2 = nn.Linear(input_size, hidden_size, bias=bias_ih)
        self.w3 = nn.Linear(input_size, hidden_size, bias=bias_ih)

    def forward(self, y, x, hx):
        # if hx is None:
        #     hx = Variable(x.data.new().resize_((x.size(0), self.hidden_size)).fill_(0))
        r = F.sigmoid(self.w1(y) + self.w2(x) + self.w3(hx))
        return r


class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, bias_ih=True, bias_hh=False):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias_ih = bias_ih
        self.bias_hh = bias_hh
        self._load_gru_cell()

    def _load_gru_cell(self):
        self.gru_cell = GRUCell(self.input_size, self.hidden_size,
                                self.bias_ih, self.bias_hh)

    def forward(self, y, x, hx, max_length=None):
        batch_size = y.size(0)
        seq_length = y.size(1)

        if max_length is None:
            max_length = seq_length
        output = []
        for i in range(max_length):
            hx = self.gru_cell(y[:, i, :], x, hx=hx)
            output.append(hx.view(batch_size, 1, self.hidden_size))
        output = torch.cat(output, 1)
        return output, hx
