from __future__ import unicode_literals, print_function, division
import random
import time

import torch.nn as nn
from torch import optim
import torch

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
from torch.autograd import Variable

SOS_token = 0
EOS_token = 1
use_cuda = False
MAX_LENGTH = 10
teacher_forcing_ratio = True
Q = 4


def trainIters(input_lang, output_lang, pairs, encoder, decoder, n_iters, print_every=1000, plot_every=100,
               learning_rate=0.001):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    for k in range(50):
        print("Epoch {}".format(k))
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
        training_pairs = [variablesFromPair(input_lang, output_lang, p) for p in pairs]
        criterion = nn.NLLLoss()

        for i in range(1, len(training_pairs) + 1):
            training_pair = training_pairs[i - 1]
            sentence = training_pair[0]
            target = training_pair[1]

            loss = train(sentence, target, encoder,
                         decoder, encoder_optimizer, decoder_optimizer,
                         criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if i % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                perc = i / len(training_pairs)
                print('%s (%d %d%%) %.4f' % (timeSince(start, i / perc),
                                             i, perc * 100, print_loss_avg))

            if i % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    showPlot(plot_losses)


def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def variableFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    # indexes = [SOS_token] * int(Q/2) + indexes + [EOS_token] * int(Q/2)
    result = Variable(torch.LongTensor(indexes + [EOS_token]).view(-1, 1))
    if use_cuda:
        return result.cuda()
    else:
        return result


def variablesFromPair(input_lang, output_lang, pair):
    sentence = variableFromSentence(input_lang, pair[0])
    target = variableFromSentence(output_lang, pair[1])
    return (sentence, target)


def train(sentence, target, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    target_length = target.size()[0]

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    loss = 0
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            enc = encoder(sentence, decoder_input, hidden)
            decoder_output, hidden = decoder(
                decoder_input, hidden, enc)
            loss += criterion(decoder_output, target[di])
            decoder_input = target[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, hidden, encoder_output, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
