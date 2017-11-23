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
    embedding_dim = 0

    # Load all glove vectors, put them in a matrix
    for line in f:
        split_line = line.split()
        word = split_line[0]
        embedding = np.array([float(val) for val in split_line[1:]])
        if embedding_dim == 0:
            embedding_dim = len(embedding)
            embeddings_matrix = np.zeros((vocab_size, embedding_dim))

        # Use only words that are in the corpus
        if word in w2i:
            embeddings_matrix[w2i[word], :] = embedding

    # Replace zero vectors with random numbers
    for i, row in enumerate(embeddings_matrix):
        if not (False in [n == 0 for n in row]):
            embeddings_matrix[i, :] = np.random.rand(1, embedding_dim)
    return embeddings_matrix


def to_indices(sequence, w2i):
    """
    Represent a history of words as a list of indices.
    """
    return [w2i[word] for word in sequence]


def to_pairs(document, summaries, size):
    document.name
    sequence = [w for s in document.sentences for w in s]
    pairs = []
    for summary in summaries:
        print(summary.name)
        summary = [w for s in summary.sentences for w in s]
        summary = ['<s>'] * (size - 1) + summary
        for i in range(size, len(summary)):
            pairs.append((sequence, summary[i - size:i], summary[i]))
    return pairs


def collection_to_pairs(documents, summaries, w2i, context_size):
    text_pairs = []
    for i, document in enumerate(documents):
        p = to_pairs(documents[i], summaries[i], context_size)
        text_pairs.extend(p)
    return text_pairs


def flatten(l):
    return [w for s in l for w in s]
