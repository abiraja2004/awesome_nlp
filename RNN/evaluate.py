from __future__ import unicode_literals, print_function, division
import random
import torch
from torch.autograd import Variable
from train import variableFromSentence

SOS_token = 0
EOS_token = 1
use_cuda = False
MAX_LENGTH = 10
teacher_forcing_ratio = 0.5


def evaluate(input_lang, output_lang, encoder, decoder, sentence,
             max_length=MAX_LENGTH):
    hidden = encoder.initHidden()
    sentence = variableFromSentence(input_lang, sentence)

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoded_words = []

    for di in range(max_length):
        enc = encoder(sentence, decoder_input, hidden)
        decoder_output, hidden = decoder(
            decoder_input, hidden, enc)
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == EOS_token:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(output_lang.index2word[ni])

        decoder_input = Variable(torch.LongTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    return decoded_words



def evaluateRandomly(input_lang, output_lang, pairs, encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words = evaluate(input_lang, output_lang, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')
