from __future__ import unicode_literals, print_function, division
import random
import torch
from torch.autograd import Variable


def evaluate(w2i, i2w, encoder, decoder, sentence,
             max_length, enable_cuda=False):
    hidden = encoder.initHidden(1)
    sentence = Variable(torch.LongTensor([sentence]))

    y = Variable(torch.LongTensor([[w2i["<s>"]]]))
    y = y.cuda() if enable_cuda else y

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
        y = y.cuda() if enable_cuda else y
    return decoded_words


def evaluateRandomly(w2i, i2w, pairs, encoder, decoder, max_length, cuda=False):
    for i in range(10):
        pair = random.choice(pairs)
        print('>', [i2w[word] for word in pair[0]])
        print('=', [i2w[word] for word in pair[1]])
        output_words = evaluate(w2i, i2w, encoder, decoder, pair[0], max_length, cuda)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
