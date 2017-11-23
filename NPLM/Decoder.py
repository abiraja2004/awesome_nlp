from torch.autograd import Variable
from utils import to_indices
import torch


class Decoder(object):
    def __init__(self, w2i, i2w, context_size):
        self.word2idx = w2i
        self.idx2word = i2w
        self.C = context_size


class Greedy_Decoder(Decoder):
    def __init__(self, w2i, i2w, context_size):
        super().__init__(w2i, i2w, context_size)

    def decode(self, sequence, model):
        summary = ['<s>'] * self.C
        sequence_i = Variable(
            torch.LongTensor(to_indices(sequence, self.word2idx))
        )

        i = self.C

        while(summary.count('</s>') < 2 and i < 100):
            summary_i = Variable(
                torch.LongTensor(to_indices(summary[i - self.C:i], self.word2idx))
            )
            scores = model.forward(sequence_i, summary_i)
            predict = scores.data.numpy().argmax(axis=1)[0]
            # print(scores, predict)
            continuation = self.idx2word[predict]
            summary.append(continuation)
            i += 1
        return(summary[self.C - 1:])
