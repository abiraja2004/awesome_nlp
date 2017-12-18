from __future__ import unicode_literals, print_function, division
from data import Gigaword_Collection
import torch
import argparse
import logging
import pickle
from collections import defaultdict
from torch.autograd import Variable


def clean(sequence):
    """Remove start and ending tags from sentence."""
    while "<s>" in sequence:
        sequence.remove("<s>")
    while "</s>" in sequence:
        sequence.remove("</s>")
    return sequence


def greedy(w2i, i2w, encoder, decoder, sentence, max_length):
    """Decode a sentence greedily."""
    hidden = encoder.initHidden(1)
    sentence = Variable(torch.LongTensor([[w2i[word] for word in sentence]]))

    y = Variable(torch.LongTensor([[w2i["<s>"]]]))

    decoded_words = []

    for di in range(max_length):
        enc, y = encoder(sentence, hidden, y)
        decoder_output, hidden = decoder(y, hidden, enc)
        topv, topi = decoder_output.data.topk(1)

        ni = topi[0][0]
        if ni == w2i["</s>"]:
            decoded_words.append('</s>')
            break
        else:
            decoded_words.append(i2w[ni])

        y = Variable(torch.LongTensor([[ni]]))
    decoder.cell.clear_m()
    return decoded_words


def beam_search(w2i, i2w, encoder, decoder, sentence, length, beam_size):
    """Given a sequence and a model, generate a summary according to
    beam search."""

    # Initalize the summary with enough starting tags
    hidden = encoder.initHidden(1)
    y = [w2i["<s>"]]
    sentence = Variable(torch.LongTensor([[w2i[word] for word in sentence]]))

    # Initialize hypotheses with three most probable words after start tags
    probs, indices, hidden = predict(encoder, decoder, sentence, y, hidden,
                                     beam_size)
    hypotheses = [(y, y + [indices[0][i]], probs[0][i], hidden)
                  for i in range(beam_size)]
    final = []

    # For every index in summary, reestimate top K best hypotheses
    for i in range(length):
        # Gather beam_size * beam_size new hypotheses
        n_h = {}
        num_hypotheses = len(hypotheses)
        for j in range(num_hypotheses):
            y, summary, prob, hidden = hypotheses[j]
            probs, indices, hidden = predict(encoder, decoder, sentence, y,
                                             hidden, beam_size)

            for k in range(len(indices[0])):
                token = indices[0][k]
                new_prob = prob + probs[0][k]

                # Only keep the best hypothesis per continuation
                if ((token not in n_h) or
                   (token in n_h and new_prob > n_h[token][2])):
                    n_h[token] = (token, summary + [token], new_prob, hidden)

        # Select top K hypotheses from new_hypotheses
        hypotheses = select_top(list(n_h.values()), beam_size)

        tmp = []
        for h in hypotheses:
            if w2i["</s>"] in h[1]:
                final.append(h)
            else:
                tmp.append(h)
        hypotheses = tmp

    hypotheses.extend(final)
    # Indices to words
    summary, prob, _ = select_top(hypotheses, 1, True)[0]
    summary = [i2w[w] for w in summary]
    decoder.cell.clear_m()
    return(summary)


def predict(encoder, decoder, sequence, y, hidden, beam_size):
    y = Variable(torch.LongTensor([y]))
    enc, y = encoder(sequence, hidden, y)
    decoder_output, hidden = decoder(y, hidden, enc)
    prob, index = decoder_output.data.topk(beam_size)
    return prob, index, hidden


def select_top(hypotheses, K, final=False):
    # Select top K hypotheses with highest probs
    if final:
        for i, (_, hypothesis, prob, hidden) in enumerate(hypotheses):
            lp = (5 + len(hypothesis))**1.5 / (5 + 1)**1.5
            hypotheses[i] = (hypothesis, prob / lp, hidden)
    return sorted(hypotheses, key=lambda x: x[1], reverse=True)[:K]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='File for testing the RAN model.')
    parser.add_argument('--decoder', type=str, default='grd',
                        help='type of decoder: grd or bms')
    parser.add_argument('--beam_size', type=int, default=3,
                        help='beam size for beam search decoder')
    parser.add_argument('--length', type=int, default=12,
                        help='desired summary length')
    parser.add_argument('--model', type=str, default='nplm_model.pt',
                        help='path for saving model')
    parser.add_argument('--save', type=str, default='summaries.txt',
                        help='file in which to save predicted summaries')
    parser.add_argument('--documents', type=str,
                        default='../sumdata/train/valid.article.txt')
    parser.add_argument('--summaries', type=str,
                        default='../sumdata/train/valid.title.txt')
    parser.add_argument('--encoder', type=str, help='Encoder torch pt file.')
    parser.add_argument('--rnn_decoder', type=str, help='Decoder torch pt file.')
    parser.add_argument('--w2i', default='models/w2i.pickle', type=str)
    parser.add_argument('--i2w', default='models/i2w.pickle', type=str)
    parser.add_argument('--nr_docs', type=int, default=10000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    # Load trained models along with dictionaries
    w2i = pickle.load(open(args.w2i, 'rb'))
    w2i = {key: value for key, value in w2i}
    i2w = pickle.load(open(args.i2w, 'rb'))
    i2w = {key: value for key, value in i2w}
    w2i = defaultdict(lambda: w2i["unk"], w2i)

    encoder = torch.load(args.encoder,
                         map_location=lambda storage, location: storage)
    decoder = torch.load(args.rnn_decoder,
                         map_location=lambda storage, location: storage)
    corpus = Gigaword_Collection(args.documents, args.summaries, args.nr_docs)

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
        if args.decoder == "grd":
            summary = greedy(w2i, i2w, encoder, decoder, doc, args.length)
        else:
            summary = beam_search(w2i, i2w, encoder, decoder, doc, args.length,
                                  args.beam_size)
        predictions.append(" ".join(clean(summary)))
        gold.append(" ".join(clean(gold_summary)))
        docs.append(" ".join(doc))
        logging.info("Creating summary for doc {} / {}.".format(i+1, nr_docs))
    open(args.save, 'w').write("\n".join(predictions))
