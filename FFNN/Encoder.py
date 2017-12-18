import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging


class BOW_Encoder(nn.Module):
    """Simple Bag of Words encoder. Does not take into account word order."""
    def __init__(self):
        super().__init__()
        logging.info("BOW Encoder initialized.")

    def encode(self, embeds_x, embeds_y, M):
        p = Variable(torch.FloatTensor([1 / M for i in range(M)]))
        enc = torch.matmul(p, embeds_x)
        return enc


class Attention_Based_Encoder(nn.Module):
    """Attention based encoder, builds upon the BOW Encoder."""
    def __init__(self, context, embed_dim, Q):
        super().__init__()
        self.Q = Q
        self.contex = context
        self.embed_dim = embed_dim
        logging.info("Attention Based Encoder initialized.")

    def encode(self, embeds_x, embeds_y, P, M):
        # x bar is a smoothed version of x tilde
        x_dim = embeds_x.size()

        x_bar = torch.FloatTensor(x_dim)

        for i in range(M):
            s = max(i - self.Q, 0)
            e = min(i + self.Q, M - 1)

            # batch mode
            if len(x_dim) == 3:
                x_bar[:, i, :] = torch.sum(
                    embeds_x.data[:, s:e, :], 1) / self.Q
            # regular mode
            elif len(x_dim) == 2:
                x_bar[i, :] = torch.sum(embeds_x.data[s:e, :], 0) / self.Q
        x_bar = Variable(x_bar)

        a = P @ embeds_y
        p = embeds_x @ a

        if len(x_dim) == 3:
            # You cannot specify an axis, so you have to transpose and
            # reverse that afterwards. Very annoying pytorch feature...
            p = (F.softmax(p.transpose(0, 1)))
            p = p.transpose(0, 1)

            x_bar = x_bar.transpose(1, 2)
            enc = (x_bar @ p).squeeze(2)
        else:
            p = p.transpose(0, 1)
            p = F.softmax(p)
            enc = (p.transpose(0, 2) @ x_bar).squeeze(0)
        return enc.squeeze(0)
