import torch
import argparse
import logging
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from random import shuffle

# Import objects and functions customized for the abstractive summarization
from NPLM import NPLM_Summarizer
from Decoder import Greedy_Decoder, Beam_Search_Decoder
from data import Collection
from utils import load_glove_matrix, to_indices, collection_to_pairs, flatten


def evaluate(model, pairs, w2i, i2w):
    """
    Evaluate a model by generating
    """
    correct = 0

    for sequence, summary, continuation in pairs:
        sequence_i = Variable(torch.LongTensor(to_indices(sequence, w2i)))
        summary_i = Variable(torch.LongTensor(to_indices(summary, w2i)))
        scores = model.forward(sequence_i, summary_i)
        predict = scores.data.numpy().argmax(axis=1)[0]

        if predict == w2i[continuation]:
            correct += 1

    return correct, len(pairs), correct/len(pairs)


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
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load data
    corpus = Collection("../opinosis/topics/", "../opinosis/summaries-gold/")
    w2i = corpus.dictionary.word2idx
    i2w = corpus.dictionary.idx2word

    train = collection_to_pairs(
        corpus.documents[:1], corpus.summaries[:1], w2i, args.context_size
    )
    # According to Jasper things mess up when you shuffle every epoch
    shuffle(train)
    logging.info("Loaded data.")
    embed = load_glove_matrix(w2i, "../glove.6B/glove.6B.300d.txt")
    logging.info("Initialized word embeddings with Glove.")

    # Initalize the network
    model = NPLM_Summarizer(args.context_size, len(w2i), len(embed[0, :]),
                            args.nhid, args.encoder, embed)
    opt = optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=1e-5)
    loss = nn.NLLLoss()
    if args.decoder == "grd":
        decoder = Greedy_Decoder(w2i, i2w, args.context_size, args.length)
    else:
        decoder = Beam_Search_Decoder(w2i, i2w, args.context_size, args.length,
                                      args.beam_size, args.verbose)

    for i in range(args.epochs):
        for j, (sequence, summary, continuation) in enumerate(train):
            # Forward pass
            sequence_i = Variable(torch.LongTensor(to_indices(sequence, w2i)))
            summary_i = Variable(torch.LongTensor(to_indices(summary, w2i)))
            scores = model.forward(sequence_i, summary_i)

            # Calculate loss
            target = Variable(torch.LongTensor([w2i[continuation]]))
            output = loss(scores, target)

            # Backward pass
            opt.zero_grad()
            model.zero_grad()
            output.backward()
            opt.step()

            # Output accuracy
            if j % 1000 == 0:
                _, _, acc = evaluate(model, train, w2i, i2w)
                print("Epoch {}, iter {}, train acc={}".format(i, j, acc))

        # Output predicted summaries to file
        s = []
        s.append("\n\nEpoch {}\n---------------".format(i))
        for k, d in enumerate(corpus.documents[:3]):
            summary = decoder.decode(torch.LongTensor(
                to_indices(flatten(d.sentences), w2i)), model
            )
            s.append("\n{}".format(d.name))
            s.append(" ".join(summary))
        open("summaries.txt", 'a').write("\n".join(s))

    torch.save(model, args.save)
