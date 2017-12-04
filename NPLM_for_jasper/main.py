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
from Decoder import Greedy_Decoder, Beam_Search_Decoder
from data import Opinosis_Collection, Gigaword_Collection
from utils import load_glove_matrix
from collections import Counter


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
    parser.add_argument('--documents', type=str, default='../opinosis/topics/',
                        help='path to documents to summarize')
    parser.add_argument('--summaries', type=str, help='path to gold summaries',
                        default='../opinosis/summaries-gold')
    parser.add_argument('--emfile', default="../glove.6B/glove.6B.300d.txt",
                        type=str, help='word embeddings file')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                        help='batch size')
    parser.add_argument('--context_size', type=int, default=5,
                        help='context size for summary window')
    parser.add_argument('--decoder', type=str, default='grd',
                        help='type of decoder: grd or bms')
    parser.add_argument('--encoder', type=str, default='att',
                        help='type of encoder: bow or att')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='beam size for beam search decoder')
    parser.add_argument('--length', type=int, default=30,
                        help='desired summary length')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--save', type=str, default='nplm_model.pt',
                        help='path for saving model')
    parser.add_argument('--verbose', type=bool, default=False,
                        help='verbose mode for beam search decoder')
    parser.add_argument('--save_summaries', type=str, default='summaries.txt',
                        help='file in which to save predicted summaries')
    parser.add_argument('--nr_docs', type=int, default=0,
                        help='number of documents to use from training set')
    args = parser.parse_args()
    if args.cuda and torch.cuda.is_available():
        cuda_enabled = True
        print("CUDA is enabled")
    else:
        cuda_enabled = False
        print("CUDA is disabled")

    logging.basicConfig(level=logging.INFO)

    # Load data
    corpus = Gigaword_Collection(args.documents, args.summaries,
                                 args.nr_docs)
    docs, train = corpus.collection_to_pairs(args.context_size)
    shuffle(train)
    w2i = corpus.dictionary.word2idx
    i2w = corpus.dictionary.idx2word
    pickle.dump(list(w2i.items()), open("models/w2i.pickle", 'wb'))
    pickle.dump(list(i2w.items()), open("models/i2w.pickle", 'wb'))
    batches = batchify(docs, train, args.batch_size)
    logging.info("Loaded data.")

    embed = load_glove_matrix(w2i, "glove.6B.300d.txt")
    logging.info("Initialized word embeddings with Glove.")

    # Initalize the network
    model = NPLM_Summarizer(args.context_size, len(w2i), len(embed[0, :]),
                            args.nhid, args.encoder, embed)
    if cuda_enabled:
        model.cuda()
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    opt = optim.Adam(params=parameters, lr=args.lr, weight_decay=1e-5)
    criterion = nn.NLLLoss()
    if args.decoder == "grd":
        decoder = Greedy_Decoder(w2i, i2w, args.context_size, args.length)
    else:
        decoder = Beam_Search_Decoder(w2i, i2w, args.context_size, args.length,
                                      args.beam_size, args.verbose)

    for i in range(args.epochs):
        for j, batch in enumerate(batches):
            # Forward pass
            filled_batch = fill_batch(docs, batch)
            sequences, summaries, continuations = filled_batch
            if cuda_enabled:
                scores = model.forward(Var(sequences).cuda(), Var(summaries).cuda(), True)
            else:
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
        torch.save(model, "models/epoch{}_".format(i) + args.save)

    logging.info("Finished training!")
    # logging.info("Finished training, new summaries are predicted.")
    # # Output predicted summaries to file
    # s = []
    # s.append("\n\nEpoch {}\n---------------".format(i))
    # for k in range(0, args.nr_docs):
    #     doc = corpus.documents[k].text
    #     gold_summary = corpus.summaries[k]
    #     summary = decoder.decode(doc, model, len(gold_summary), False)
    #     s.append("predicted summary: " + " ".join(summary))
    #     s.append("gold summary: " + " ".join(gold_summary) + "\n")
    # open(args.save_summaries, 'a').write("\n".join(s))
