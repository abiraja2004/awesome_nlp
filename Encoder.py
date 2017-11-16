import numpy as np
import nltk
from utils import get_w2i, load_glove_matrix


class Encoder:
    """General class for encoders."""
    def __init__(self, w2i, embeddings):
        self.F = embeddings
        self.w2i = w2i


class BOW_Encoder(Encoder):
    """Simple Bag of Words encoder. Does not take into account word order."""
    def __init__(self, w2i, embeddings):
        super().__init__(w2i, embeddings)

    def encode(self, sequence):
        x_tilde = np.matrix([self.F[self.w2i[word], :] for word in sequence])
        M = len(sequence)
        p = [1/M for i in range(M)]
        return np.matmul(np.transpose(p), x_tilde)


if __name__ == "__main__":
    words = [word.lower() for word in nltk.corpus.treebank.words()]
    i2w, w2i = get_w2i(words)
    embeddings_matrix = load_glove_matrix(w2i, "glove.6B/glove.6B.50d.txt")

    bow_enc = Encoder(w2i, embeddings_matrix)
    bow_enc.encode(["What", "are", "you", "doing", "?"])
