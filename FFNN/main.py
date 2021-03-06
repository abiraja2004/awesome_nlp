import torch
import argparse
import logging
import torch.nn as nn
import pickle
import torch.optim as optim
from torch.autograd import Variable as Var
from torch import LongTensor as LT
from random import shuffle

# Import objects and functions customized for the abstractive summarization
from NPLM import NPLM_Summarizer
from data import Gigaword_Collection
from utils import load_glove_matrix


def evaluate(model, docs, pairs):
    """
    Evaluate a model by generating
    """
    correct = 0

    for index, _, summary, continuation in pairs:
        scores = model.forward(Var(LT(docs[index])), Var(LT(summary)), False)
        predict = scores.data.numpy().argmax(axis=1)[0]
        if predict == int(continuation[0]):
            correct += 1

    return correct, len(pairs), correct / len(pairs)


def batchify(docs, pairs, batch_size):
    """
    Separate training samples in sets of equal sequence length.
    Do not "fill" the document (the sentence to summarize) yet,
    to spare memory.
    """
    batches = []
    lengths = [pair[1] for pair in pairs]
    for length in set(lengths):
        pairs_with_length = [pair for pair in pairs if pair[1] == length]
        for i in range(batch_size, len(pairs_with_length), batch_size):
            batches.append(pairs_with_length[i - batch_size:i])
    return batches


def fill_batch(docs, batch):
    """
    Turn data from training samples into a batch that can be read by the
    neural network.
    """
    sequences = []
    summaries = []
    continuations = []
    for index, _, summary, continuation in batch:
        sequences.append(docs[index])
        summaries.append(summary)
        continuations.append(continuation)
    return LT(sequences), LT(summaries), LT(continuations).squeeze(1)


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
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=10,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--context_size', type=int, default=5,
                        help='context size for summary window')
    parser.add_argument('--encoder', type=str, default='att',
                        help='type of encoder: bow or att')
    parser.add_argument('--beam_size', type=int, default=5,
                        help='beam size for beam search decoder')
    parser.add_argument('--length', type=int, default=30,
                        help='desired summary length')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--save', type=str, default='nplm_model.pt',
                        help='path for saving model')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='verbose mode for beam search decoder')
    parser.add_argument('--nr_docs', type=int, default=0,
                        help='number of documents to use from training set')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load data
    corpus = Gigaword_Collection(args.documents, args.summaries,
                                 args.nr_docs)
    docs, train = corpus.collection_to_pairs(args.context_size)
    shuffle(train)
    w2i = corpus.dictionary.word2idx
    i2w = corpus.dictionary.idx2word
    pickle.dump(list(w2i.items()), open(args.save + "w2i.pickle", 'wb'))
    pickle.dump(list(i2w.items()), open(args.save + "i2w.pickle", 'wb'))
    batches = batchify(docs, train, args.batch_size)
    logging.info("Loaded data.")

    embed = load_glove_matrix(w2i, args.emfile)
    logging.info("Initialized word embeddings with Glove.")

    # Initalize the network
    model = NPLM_Summarizer(args.context_size, len(w2i), len(embed[0, :]),
                            args.nhid, args.encoder, embed)
    opt = optim.SGD(params=model.parameters(), lr=args.lr)
    criterion = nn.NLLLoss()

    for i in range(args.epochs):
        for j, batch in enumerate(batches):
            # Forward pass
            filled_batch = fill_batch(docs, batch)
            sequences, summaries, continuations = filled_batch

            scores = model.forward(Var(sequences), Var(summaries), True)
            logging.debug("Epoch {}, iter {}. Forward pass done.".format(i, j))

            # Calculate loss
            output = criterion(scores, Var(continuations))
            logging.debug("Epoch {}, iter {}. Calculated loss.".format(i, j))

            # Backward pass
            opt.zero_grad()
            model.zero_grad()
            output.backward()
            opt.step()
            logging.debug("Epoch {}, iter {}.Updated parameters.".format(i, j))

            # Output accuracy
            if j % 10000 == 0:
                _, _, acc = evaluate(model, docs, train[:100])
                print("Epoch {}, iter {}, train acc={}".format(i, j, acc))
    torch.save(model, args.save + "epoch{}.pt".format(i))

    logging.info("Finished training!")
