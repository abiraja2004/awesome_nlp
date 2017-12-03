import nltk
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from utils import load_glove_matrix
import logging
import numpy as np


class BOW_Encoder():
    """Simple Bag of Words encoder. Does not take into account word order."""
    def __init__(self):
        logging.info("BOW Encoder initialized.")

    def encode(self, embeds_x, embeds_y, M):
        p = Variable(torch.FloatTensor([1 / M for i in range(M)]))
        enc = torch.matmul(p, embeds_x)
        return enc


class Attention_Based_Encoder():
    """Attention based encoder, builds upon the BOW Encoder."""
    def __init__(self, context, embed_dim, Q):
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

        a = P @ embeds_y.t()
        p = embeds_x @ a
        p = F.softmax(p.t())
        enc = p @ x_bar
        return enc
