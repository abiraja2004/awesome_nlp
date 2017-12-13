from __future__ import unicode_literals, print_function, division
from evaluate import evaluateRandomly
from data import Gigaword_Collection
import torch
import argparse
import logging
import pickle
import torch.nn as nn
from collections import defaultdict
from torch.autograd import Variable

# Import objects and functions customized for the abstractive summarization


def clean(sequence):
    while "<s>" in sequence:
        sequence.remove("<s>")
    while "</s>" in sequence:
        sequence.remove("</s>")
    return sequence


def evaluate(w2i, i2w, encoder, decoder, sentence,
             max_length, cuda=False):
    hidden = encoder.initHidden(1)
    sentence = Variable(torch.LongTensor([[i2w[word] for word in sentence]]))

    decoder_input = Variable(torch.LongTensor([[w2i["<s>"]]]))
    decoder_input = decoder_input.cuda() if cuda else decoder_input

    decoded_words = []

    for di in range(max_length):
        enc = encoder(sentence, hidden)
        decoder_output, hidden = decoder(decoder_input, hidden, enc)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == w2i["</s>"]:
            decoded_words.append('</s>')
            break
        else:
            decoded_words.append(i2w[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if cuda else decoder_input
    return decoded_words


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='NPLM Language Model for abstractive summarization.')
    parser.add_argument('--decoder', type=str, default='grd',
                        help='type of decoder: grd or bms')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='beam size for beam search decoder')
    parser.add_argument('--length', type=int, default=30,
                        help='desired summary length')
    parser.add_argument('--model', type=str, default='nplm_model.pt',
                        help='path for saving model')
    parser.add_argument('--save_summaries', type=str, default='summaries.txt',
                        help='file in which to save predicted summaries')
    parser.add_argument('--documents', type=str,
                        default='../sumdata/train/valid.article.txt')
    parser.add_argument('--summaries', type=str,
                        default='../sumdata/train/valid.title.txt')
    parser.add_argument('--w2i', default='models/w2i.pickle', type=str)
    parser.add_argument('--i2w', default='models/i2w.pickle', type=str)
    parser.add_argument('--verbose', type=bool, default=False)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    w2i = pickle.load(open(args.w2i, 'rb'))
    w2i = {key: value for key, value in w2i}
    i2w = pickle.load(open(args.i2w, 'rb'))
    i2w = {key: value for key, value in i2w}
    w2i = defaultdict(lambda: w2i["unk"], w2i)
    # model = torch.load(args.model, map_location=lambda storage, location: storage)
    encoder = pickle.load(open("models/epoch0_enc.pickle", 'rb'))
    decoder = pickle.load(open("models/epoch0_dec.pickle", 'rb'))
    corpus = Gigaword_Collection(args.documents, args.summaries, args.nr_docs,
                                 False)

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
        summary = evaluate(w2i, i2w, encoder, decoder, doc, args.length)
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
