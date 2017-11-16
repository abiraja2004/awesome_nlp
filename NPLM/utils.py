from collections import defaultdict
import numpy as np


def get_w2i(corpus):
    # Give every word an index
    w2i = defaultdict(lambda: len(w2i))
    i2w = dict()
    UNK = w2i["<unk>"]
    for w in corpus:
        w2i[w]
        i2w[w2i[w]] = w
    w2i = defaultdict(lambda: UNK, w2i)
    return i2w, w2i


def load_glove_matrix(w2i, glove_file):
    """
    Represent word embeddings in a matrix to initialize the nn's embeddings.
    """
    f = open(glove_file, 'rb')
    vocab_size = len(w2i)
    embedding_dim = 50
    embeddings_matrix = np.zeros((vocab_size, embedding_dim))

    # Load all glove vectors, put them in a matrix
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])

        # Use only words that are in the corpus
        if word in w2i:
            embeddings_matrix[w2i[word], :] = embedding

    # Replace zero vectors with random numbers
    for i, row in enumerate(embeddings_matrix):
        if not (False in [n == 0 for n in row]):
            embeddings_matrix[i, :] = np.random.rand(1, embedding_dim)
    return embeddings_matrix
