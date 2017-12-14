from __future__ import unicode_literals, print_function, division
from random import random
import time
import logging
import torch
import pickle
import os
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch import LongTensor as LT


def trainIters(batches, w2i, encoder, decoder, epochs, learning_rate,
               max_length, teacher_forcing_ratio, enable_cuda=False):
    start = time.time()
    plot_losses = []
    enc_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    dec_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for i in range(epochs):
        total_loss = 0
        # for k, filename in enumerate(batch_files):
        #     batches = pickle.load(open("batches/" + filename, 'rb'))
        n = len(batches)
        for j, (sentence, target) in enumerate(batches):
            if enable_cuda:
                loss = train(sentence.cuda(), target.cuda(), encoder, decoder,
                             enc_optimizer, dec_optimizer, criterion,
                             max_length, w2i, teacher_forcing_ratio, enable_cuda)
            else:
                loss = train(sentence, target, encoder, decoder, enc_optimizer,
                             dec_optimizer, criterion, max_length, w2i,
                             teacher_forcing_ratio, enable_cuda)
            total_loss += loss
            if j % 3000 == 0:
                torch.save(encoder, os.path.dirname(os.path.realpath(__file__)) + "/models/epoch{}_batch{}_enc.pt".format(i, j))
                torch.save(decoder, os.path.dirname(os.path.realpath(__file__)) + "/models/epoch{}_batch{}_dec.pt".format(i, j))
            logging.info("Epoch {}, batch {}/{}, average loss {}".format(
                i+1, j+1, n, total_loss/(j+1))
            )
        torch.save(encoder, os.path.dirname(os.path.realpath(__file__)) + "/models/epoch{}_enc.pt".format(i))
        torch.save(decoder, os.path.dirname(os.path.realpath(__file__)) + "/models/epoch{}_dec.pt".format(i))
        plot_losses.append(total_loss)

    end = time.time()
    logging.info("Training took {} seconds.".format(end-start))
    showPlot(plot_losses)


def train(sentence, target, encoder, decoder, enc_optimizer, dec_optimizer,
          criterion, max_length, w2i, teacher_forcing_ratio, enable_cuda=False):
    hidden = encoder.initHidden(sentence.size()[0])

    enc_optimizer.zero_grad()
    dec_optimizer.zero_grad()

    target_length = target.size()[1]

    y = target[:, 0]
    y = y.cuda() if enable_cuda else y

    loss = 0
    use_teacher_forcing = True if random() < teacher_forcing_ratio else False
    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for i in range(1, target_length):
            enc, y = encoder(sentence, hidden, y)
            prediction, hidden = decoder(y, hidden, enc)
            if enable_cuda:
                loss += criterion(prediction, target[:, i]).cuda()
            else:
                loss += criterion(prediction, target[:, i])
            y = target[:, i]  # Teacher forcing
            y = y.cuda() if enable_cuda else y
    else:
        # Without teacher forcing: use its own predictions as the next input
        for i in range(1, target_length):
            enc, y = encoder(sentence, hidden, y)
            prediction, hidden = decoder(y, hidden, enc)
            topv, y = prediction.transpose(0, 1).topk(1, dim=0)
            y = y.squeeze(0).cuda() if enable_cuda else y.squeeze(0)
            if enable_cuda:
                loss += criterion(prediction, target[:, i]).cuda()
            else:
                loss += criterion(prediction, target[:, i])

    loss.backward(retain_graph=True)
    enc_optimizer.step()
    dec_optimizer.step()

    return loss.data[0] / target_length


def showPlot(points):
    plt.figure()
    plt.plot(points)
    plt.show()
