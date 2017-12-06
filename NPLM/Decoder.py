import torch
import logging
from torch.autograd import Variable
from torch import LongTensor as LT


class Decoder(object):
    def __init__(self, word2idx, idx2word, context_size, length):
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.C = context_size
        self.length = length


class Greedy_Decoder(Decoder):
    """
    The Greedy Decoder simply selects the word with the highest probability
    as the next word.
    """
    def __init__(self, w2i, i2w, context_size, length):
        super().__init__(w2i, i2w, context_size, length)
        logging.info("Greedy Decoder initialized.")

    def decode(self, sequence, model, len, sentences=False):
        """Given a sequence and a model, generate a summary in a
        greedy manner."""

        # Initalize the sentence with enough starting tags
        summary = [self.word2idx['<s>']] * self.C
        sequence = Variable(LT([self.word2idx[w] for w in sequence]))

        # Greedily select the word with the highest probability
        for i in range(self.C, len+self.C-1):
            summary = self.find_next_word(summary, i, model, sequence)
            if self.word2idx['</s>'] in summary:
                break

        # Indices to words
        summary = [self.idx2word[w] for w in summary[self.C-1:]]
        return(summary)

    def find_next_word(self, summary, i, model, sequence):
        # Ensure that we do not predict unk
        summary_i = Variable(LT(summary[i - self.C:i]))
        scores = model.forward(sequence, summary_i, False)
        prob, index = torch.topk(scores.data, 2)
        if self.idx2word[index[0][0]] == "unk":
            summary.append(index[0][1])
        else:
            summary.append(index[0][0])
        return summary


class Beam_Search_Decoder(Decoder):
    """
    The Beam Search Decoder maintains K current best hypotheses at
    a time.
    """
    def __init__(self, w2i, i2w, context_size, length, beam_size, verbose=False):
        super().__init__(w2i, i2w, context_size, length)
        self.beam_size = beam_size
        logging.info("Beam Search Decoder initialized.")
        self.verbose = verbose

    def decode(self, sequence, model, length, sentences=False):
        """Given a sequence and a model, generate a summary according to
        beam search."""

        # Initalize the summary with enough starting tags
        summary = [self.word2idx['<s>']] * self.C
        sequence = Variable(LT([self.word2idx[w] for w in sequence]))

        # Initialize hypotheses with three most probable words after start tags
        probs, indices = self.predict(model, sequence, summary[:self.C])
        hypotheses = [(summary + [indices[0][i]], probs[0][i])
                      for i in range(self.beam_size)]
        final = []
        if self.verbose:
            print("Top K")
            for hypothesis in hypotheses:
                self.print_hypothesis(hypothesis)

        # For every index in summary, reestimate top K best hypotheses
        for i in range(self.C+1, length+self.C-1):
            # Gather beam_size * beam_size new hypotheses
            n_h = {}
            num_hypotheses = len(hypotheses)
            for j in range(num_hypotheses):
                hypothesis, prob = hypotheses[j]
                y_c = hypothesis[i - self.C:i]
                probs, indices = self.predict(model, sequence, y_c)

                for k in range(self.beam_size):
                    token = indices[0][k]
                    new_prob = prob + probs[0][k]

                    # Only keep the best hypothesis per continuation
                    if ((token not in n_h) or
                            (token in n_h and new_prob > n_h[token][1])):
                        n_h[token] = (hypothesis + [token], new_prob)

            # Select top K hypotheses from new_hypotheses
            hypotheses = self.select_top(list(n_h.values()), self.beam_size)
            if self.verbose:
                print("New Hypotheses")
                for key in n_h:
                    self.print_hypothesis(n_h[key])
                print("Top K")
                for hypothesis in hypotheses:
                    self.print_hypothesis(hypothesis)

            for h in hypotheses:
                tmp = []
                if self.word2idx["</s>"] in h:
                    final.append(h)
                    self.beam_size = self.beam_size - 1
                else:
                    tmp.append(h)
                hypotheses = tmp

            if self.beam_size == 0 or not hypotheses:
                break

        hypotheses.extend(final)

        # Indices to words
        summary, prob = self.select_top(hypotheses, 1)[0]
        summary = [self.idx2word[w] for w in summary[self.C - 1:]]
        return(summary)

    def predict(self, model, sequence, summary):
        summary_i = Variable(torch.LongTensor(summary))
        scores = model.forward(sequence, summary_i, False)
        prob, index = torch.topk(scores.data, self.beam_size)
        return prob, index

    def select_top(self, hypotheses, K):
        # Ensure that we do not predict unk
        to_delete = []
        for i, h in enumerate(hypotheses):
            if self.idx2word[h[0][-1]] == "unk":
                to_delete.append(i)

        # do not delete hypotheses if all predict unk
        if len(hypotheses) != len(to_delete):
            for index in to_delete:
                hypotheses.pop(index)

        # Select top K hypotheses with highest probs
        return sorted(hypotheses, key=lambda x: x[1], reverse=True)[:K]

    def print_hypothesis(self, hypothesis):
        hypothesis, prob = hypothesis
        hypothesis = [self.idx2word[i] for i in hypothesis]
        print("{} with probability {}".format(" ".join(hypothesis), prob))
