import torch
import argparse
import logging
import torch.nn as nn
import pickle
import torch.optim as optim
from torch.autograd import Variable as Var
from torch import LongTensor as LT
from random import shuffle
from collections import defaultdict

# Import objects and functions customized for the abstractive summarization
from NPLM import NPLM_Summarizer
from Decoder import Greedy_Decoder, Beam_Search_Decoder
from data import Gigaword_Collection


def clean(sequence):
    while "<s>" in sequence:
        sequence.remove("<s>")
    while "</s>" in sequence:
        sequence.remove("</s>")
    return sequence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='NPLM Language Model for abstractive summarization.')
    parser.add_argument('--decoder', type=str, default='grd',
                        help='type of decoder: grd or bms')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='beam size for beam search decoder')
    parser.add_argument('--length', type=int, default=12,
                        help='desired summary length')
    parser.add_argument('--model', type=str, default='nplm_model.pt',
                        help='path for saving model')
    parser.add_argument('--save_summaries', type=str, default='summaries.txt',
                        help='file in which to save predicted summaries')
    parser.add_argument('--documents', type=str,
                        default='../sumdata/train/valid.article.txt')
    parser.add_argument('--summaries', type=str,
                        default='../sumdata/train/valid.title.txt')
    parser.add_argument('--context_size', type=int, default=5)
    parser.add_argument('--nr_docs', type=int, default=0)
    parser.add_argument('--w2i', type=str)
    parser.add_argument('--i2w', type=str)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    w2i = pickle.load(open(args.w2i, 'rb'))
    w2i = {key: value for key, value in w2i}
    i2w = pickle.load(open(args.i2w, 'rb'))
    i2w = {key: value for key, value in i2w}
    w2i = defaultdict(lambda: w2i["unk"], w2i)
    print(len(w2i))
    model = torch.load(args.model, map_location=lambda storage, location: storage)
    print(model.encoder)
    corpus = Gigaword_Collection(args.documents, args.summaries, args.nr_docs,
                                 False)

    logging.info("Initialized word embeddings with Glove.")

    # Initalize the network

    criterion = nn.NLLLoss()
    if args.decoder == "grd":
        decoder = Greedy_Decoder(w2i, i2w, args.context_size, args.length)
    else:
        decoder = Beam_Search_Decoder(w2i, i2w, args.context_size, args.length,
                                      args.beam_size, args.verbose)

    # Output predicted summaries to file

    predictions = []
    gold = []

    if args.nr_docs <= 0:
        nr_docs = len(corpus.documents)
    else:
        nr_docs = args.nr_docs

    docs = []
    for i in range(nr_docs):
        doc = corpus.prepare(corpus.documents[i])
        gold_summary = corpus.prepare(corpus.summaries[i])
        summary = decoder.decode(doc, model, False)
        #print(summary)
        predictions.append(" ".join(clean(summary)))
        gold.append(" ".join(clean(gold_summary)))
        docs.append(" ".join(doc))
        logging.info("Creating summary for doc {} / {}.".format(i+1, nr_docs))
        # s.append("gold summary: " + " ".join(gold_summary) + "\n")
        if args.verbose:
            print(doc)
            print("=", gold_summary)
            print("<", summary)
            print()
    open(args.save_summaries, 'w').write("\n".join(predictions))
    open("gold.txt", 'w').write("\n".join(gold))
    open("docs.txt", 'w').write("\n".join(docs))
