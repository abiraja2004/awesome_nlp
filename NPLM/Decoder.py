from torch.autograd import Variable
from utils import to_indices
import torch


class Decoder(object):
    def __init__(self, w2i, i2w, context_size, length):
        self.word2idx = w2i
        self.idx2word = i2w
        self.C = context_size
        self.length = length


class Greedy_Decoder(Decoder):
    """The Greedy Decoder simply selects the word with the highest probability
    as the next word."""
    def __init__(self, w2i, i2w, context_size, length):
        super().__init__(w2i, i2w, context_size, length)

    def decode(self, sequence, model):
        # Initalize the sentence with enough starting tags
        summary = [self.word2idx['<s>']] * self.C
        sequence_i = Variable(
            torch.LongTensor(to_indices(sequence, self.word2idx))
        )

        i = self.C

        # Greedily select the word with the highest probability
        end_tag = self.word2idx['</s>']
        while(summary.count(end_tag) < 2 and i < 100):
            summary_i = Variable(torch.LongTensor(summary[i-self.C:i]))
            scores = model.forward(sequence_i, summary_i)
            prob, index = torch.topk(scores.data, 1)
            summary.append(index[0][0])
            i += 1

        # Indices to words
        summary = [self.idx2word[w] for w in summary[self.C - 1:]]
        return(summary)


class Beam_Search_Decoder(Decoder):
    """The Beam Search Decoder maintains K current best hypotheses at
    the time."""
    def __init__(self, w2i, i2w, context_size, beam_size, length):
        super().__init__(w2i, i2w, context_size, length)
        self.beam_size = beam_size

    def decode(self, sequence, model):
        # Initalize the summary with enough starting tags
        summary = [self.word2idx['<s>']] * self.C
        sequence = Variable(sequence)

        # Initialize hypotheses with three most probable words after start tags
        probs, indices = self.predict(model, sequence, summary[:self.C])
        hypotheses = [(summary + [indices[0][i]], probs[0][i])
                      for i in range(self.beam_size)]
        # print("Top K")
        # for hypothesis in hypotheses:
        #     self.print_hypothesis(hypothesis)

        # For every index in summary, reestimate top K best hypotheses
        for i in range(self.C+1, self.length):
            # Gather beam_size * beam_size new hypotheses
            n_h = {}
            for j in range(self.beam_size):
                hypothesis, prob = hypotheses[j]
                y_c = hypothesis[i-self.C:i]
                probs, indices = self.predict(model, sequence, y_c)

                for k in range(self.beam_size):
                    token = indices[0][k]
                    new_prob = prob + probs[0][k]
                    if ((token not in n_h) or
                       (token in n_h and new_prob > n_h[token][1])):
                        n_h[token] = (hypothesis + [token], new_prob)

            # print("New Hypotheses")
            # for key in n_h:
            #     self.print_hypothesis(n_h[key])
            # Select top K hypotheses from new_hypotheses
            hypotheses = self.select_top(list(n_h.values()), self.beam_size)
            # print("Top K")
            # for hypothesis in hypotheses:
            #     self.print_hypothesis(hypothesis)

        # Indices to words
        summary, prob = self.select_top(hypotheses, 1)[0]
        summary = [self.idx2word[w] for w in summary[self.C - 1:]]
        return(summary)

    def predict(self, model, sequence, summary):
        summary_i = Variable(torch.LongTensor(summary))
        scores = model.forward(sequence, summary_i)
        prob, index = torch.topk(scores.data, self.beam_size)
        return prob, index

    def select_top(self, list, top):
        return sorted(list, key=lambda x: x[1], reverse=True)[:top]

    def print_hypothesis(self, hypothesis):
        hypothesis, prob = hypothesis
        hypothesis = [self.idx2word[i] for i in hypothesis]
        print("{} with probability {}".format(" ".join(hypothesis), prob))
