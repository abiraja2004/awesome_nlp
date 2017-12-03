import numpy as np


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
            vec = np.random.rand(1, embedding_dim)
            embeddings_matrix[i, :] = vec / np.linalg.norm(vec)
    return embeddings_matrix
