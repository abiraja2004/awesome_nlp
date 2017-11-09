from collections import defaultdict
import time
import random
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import nltk.corpus

# Default settings, may be overwritten because of data used
CONTEXT_SIZE = 3
VOCABULARY_DIM = 40000
EMBEDDING_DIM = 50
ITER = 10


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
    f = open(glove_file, 'r')
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


def encode_history(history, w2i):
    """
    Represent a history of words as a list of indices.
    """
    return [w2i[word] for word in history]


def evaluate(model, ngrams, w2i, i2w):
    """
    Evaluate a model on a data set.
    """
    correct = 0

    for history, continuation in ngrams:
        indices = encode_history(history, w2i)
        lookup_tensor = Variable(torch.LongTensor(indices))
        scores = model(lookup_tensor)
        predict = scores.data.numpy().argmax(axis=1)[0]
        # print(history, continuation, i2w[predict])

        if predict == w2i[continuation]:
            correct += 1

    return correct, len(ngrams), correct/len(ngrams)


class NPLM(nn.Module):
    """
    Neural Probabilistic Language Model, as described by Bengio (2003).
    """
    def __init__(self, context, vocab_dim, embed_dim, pretrained=None):
        super(NPLM, self).__init__()
        self.embeddings = nn.Embedding(vocab_dim, embed_dim)

        # Use pretrained weights, a numpy matrix of shape vocab_dim x embed_dim
        if pretrained is not None:
            self.embeddings.weight.data.copy_(torch.from_numpy(pretrained))
        self.linear1 = nn.Linear(context * embed_dim, 100)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(100, vocab_dim)
        self.softmax = nn.Softmax()

    def forward(self, inputs):
        embeds = self.embeddings(inputs).view((1, -1))
        out = self.tanh(self.linear1(embeds))
        out = self.linear2(out)
        return self.softmax(out)


def to_ngrams(sentences, history_size):
    """
    Extracts all n-grams from a list of sentences.
    """
    ngrams = []
    for s in sentences:
        for i in range(len(s)-history_size):
            ngrams.append((s[i:i+history_size], s[i+history_size]))
    return ngrams


# Get corpus, word indices and ngrams for evaluation
words = [word.lower() for word in nltk.corpus.brown.words()]
sentences = [[word.lower() for word in s] for s in nltk.corpus.brown.sents()]
i2w, w2i = get_w2i(words)
print("Done reading data.")
ngrams = to_ngrams(sentences, CONTEXT_SIZE)
train = ngrams[:5000]
test = ngrams[-1000:]
print("Prepared n-grams for evaluation.")

# Initialize word embeddings with 50d glove vectors
embeddings_matrix = load_glove_matrix(w2i, "glove.6B/glove.6B.50d.txt")
print("Done with reading in glove vectors.")

VOCABULARY_DIM = len(w2i)
EMBEDDING_DIM = len(embeddings_matrix[0, :])

# Initalize the neural network
model = NPLM(CONTEXT_SIZE, VOCABULARY_DIM, EMBEDDING_DIM, embeddings_matrix)
print("Initalized the Neural Probabilistic Language Model.")
print(model)
optimizer = optim.SGD(model.parameters(), lr=0.00001)

for i in range(ITER):
    random.shuffle(train)
    train_loss = 0.0
    start = time.time()

    for j, (history, continuation) in enumerate(train):

        # forward pass
        optimizer.zero_grad()
        indices = encode_history(history, w2i)
        lookup_tensor = Variable(torch.LongTensor(indices))
        scores = model(lookup_tensor)

        loss = nn.CrossEntropyLoss()
        target = Variable(torch.LongTensor([w2i[continuation]]))
        output = loss(scores, target)
        train_loss += output.data[0]

        # backward pass
        output.backward()
        optimizer.step()

        if j % 1000 == 0:
            _, _, acc = evaluate(model, train, w2i, i2w)
            print("Epoch {}, iter {}, train acc={}".format(i, j, acc))
