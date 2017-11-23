import nltk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils import get_w2i, load_glove_matrix


class BOW_Encoder():
    """Simple Bag of Words encoder. Does not take into account word order."""
    def __init__(self):
        print("BOW Encoder initialized.")

    def encode(self, embeds_x, embeds_y, M):
        p = Variable(torch.FloatTensor([1/M for i in range(M)]))
        enc = torch.matmul(p, embeds_x)
        return enc


class Attention_Based_Encoder():
    """Attention based encoder, builds upon the BOW Encoder."""
    def __init__(self, context, embed_dim, Q):
        self.Q = Q
        self.contex = context
        self.embed_dim = embed_dim
        print("Attention Based Encoder initialized.")

    def encode(self, embeds_x, embeds_y, P, M):
        # x bar is a smoothed version of x tilde
        x_bar = torch.FloatTensor(M, self.embed_dim)
        for i in range(M):
            start = max(i-self.Q, 0)
            end = min(i+self.Q, M-1)
            x_bar[i, :] = torch.sum(embeds_x.data[start:end, :], 0) / self.Q
        x_bar = Variable(x_bar)

        p = embeds_x @ P @ embeds_y.t()
        p = torch.unsqueeze(F.softmax(p.squeeze()), 0)
        enc = p @ x_bar
        return enc


if __name__ == "__main__":
    words = [word.lower() for word in nltk.corpus.treebank.words()]
    i2w, w2i = get_w2i(words)
    embeddings_matrix = load_glove_matrix(w2i, "../glove.6B/glove.6B.50d.txt")

    bow_enc = BOW_Encoder(w2i, embeddings_matrix)
    enc = bow_enc.encode(["What", "are", "you", "doing", "?"], [])
    print(enc)
