import os
from train import trainIters
from evaluate import evaluateRandomly
from Decoder import Elman_Decoder
from Encoder import Attentive_Encoder
from data import Gigaword_Collection
from random import shuffle
from torch.autograd import Variable
import torch
import numpy as np
import argparse
import logging
import pickle


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


def batchify(pairs, w2i, batch_size):
    """
    Separate training samples in sets of equal sequence length.
    Do not "fill" the document (the sentence to summarize) yet,
    to spare memory.
    """
    batches = []
    lengths = [len(pair[0]) for pair in pairs]
    for length in set(lengths):
        pairs_with_length = [pair for pair in pairs if len(pair[0]) == length]
        for i in range(batch_size, len(pairs_with_length), batch_size):
            batches.append(pairs_with_length[i - batch_size:i])

    for i, batch in enumerate(batches):
        documents, summaries = zip(*batch)
        documents = list(documents)
        summaries = list(summaries)
        longest = max(len(l) for l in summaries)
        for j, summary in enumerate(summaries):
            diff = longest - len(summary)
            summary = summary + [w2i["</s>"]] * diff
            summaries[j] = summary
        batches[i] = (Variable(torch.LongTensor(documents)),
                      Variable(torch.LongTensor(summaries)))
    return batches


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='NPLM Language Model for abstractive summarization.')
    parser.add_argument('--documents', type=str,
                        default='../sumdata/train/train.article.txt',
                        help='path to documents to summarize')
    parser.add_argument('--summaries', type=str, help='path to gold summaries',
                        default='../sumdata/train/train.title.txt')
    parser.add_argument('--emfile', default="../glove.6B/glove.6B.200d.txt",
                        type=str, help='word embeddings file')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--length', type=int, default=30,
                        help='desired summary length')
    parser.add_argument('--enable_cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--save', type=str, default='nplm_model.pt',
                        help='path for saving model')
    parser.add_argument('--nr_docs', type=int, default=0,
                        help='number of documents to use from training set')
    parser.add_argument('--q', type=int, default=3, help='setting for encoder')
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='ratio for teacher forcing mechanism')
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG)

    if args.enable_cuda and torch.cuda.is_available():
        enable_cuda = True
        logging.info("CUDA is enabled")
    else:
        enable_cuda = False
        logging.info("CUDA is disabled")

    corpus = Gigaword_Collection(args.documents, args.summaries, args.nr_docs)
    train = corpus.collection_to_pairs()
    shuffle(train)
    w2i = corpus.dictionary.word2idx
    i2w = corpus.dictionary.idx2word

    pickle.dump(list(w2i.items()), open(os.path.dirname(os.path.realpath(__file__)) + "/models/w2i.pickle", 'wb'))
    pickle.dump(list(i2w.items()), open(os.path.dirname(os.path.realpath(__file__)) + "/models/i2w.pickle", 'wb'))

    batches = batchify(train, w2i, args.batch_size)
    pickle.dump(batches, open(os.path.dirname(os.path.realpath(__file__)) + "/batches.pickle", 'wb'))
    logging.info("Loaded data.")

    # embed = load_glove_matrix(w2i, args.emfile)
    logging.info("Initialized embeddings.")

    dim = 200
    encoder = Attentive_Encoder(len(w2i), dim, args.q, None, enable_cuda)
    decoder = Elman_Decoder(dim, len(w2i), None, args.length, enable_cuda)

    if enable_cuda:
        encoder.cuda()
        decoder.cuda()

    logging.info("Training will start shortly.")

    trainIters(batches, w2i, encoder, decoder, args.epochs, args.lr,
               args.length, args.ratio, enable_cuda)

    evaluateRandomly(w2i, i2w, train, encoder, decoder, args.length, enable_cuda)
