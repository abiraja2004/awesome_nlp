import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from Encoder import BOW_Encoder, Attention_Based_Encoder


class NPLM(nn.Module):
    """
    Neural Probabilistic Language Model, as described by Bengio (2003).
    """
    def __init__(self, context, vocab_dim, embed_dim, hidden, pretrained=None):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_dim, embed_dim)

        # Use pretrained weights, a numpy matrix of shape vocab_dim x embed_dim
        if pretrained is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pretrained))
        self.linear1 = nn.Linear(context * embed_dim, hidden)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(hidden, vocab_dim)

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.tanh(self.linear1(embeds))
        return F.log_softmax(self.linear2(out))


class NPLM_Summarizer(nn.Module):
    """
    Neural Probabilistic Language Model for abstractive summarization,
    as described by Rush (2015).
    """
    def __init__(self, context, vocab_dim, embed_dim, hidden, enc, embed=None):
        super().__init__()
        self.embeddings = nn.Embedding(vocab_dim, embed_dim)

        # Use pretrained weights, a numpy matrix of shape vocab_dim x embed_dim
        if embed is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(embed))
        self.U = nn.Linear(context * embed_dim, hidden)
        self.V = nn.Linear(hidden, vocab_dim)
        self.W = nn.Linear(embed_dim, vocab_dim)

        if enc == "bow":
            self.encoder = BOW_Encoder()
        elif enc == "att":
            P = torch.FloatTensor(torch.randn(embed_dim, context*embed_dim))
            self.P = nn.Parameter(P)
            self.encoder = Attention_Based_Encoder(context, embed_dim, 2)

        logging.info("Feedforward neural language model initialized.")

    def forward(self, x, y):
        # Use embeddings to represent input as matrices
        embeds_x = self.embeddings(x)
        embeds_y = self.embeddings(y).view((1, -1))

        # Call encoder
        enc = self.encoder.encode(embeds_x, embeds_y, self.P, len(x))

        out = F.tanh(self.U(embeds_y))
        final = torch.add(self.V(out), self.W(enc))
        return F.log_softmax(final)
