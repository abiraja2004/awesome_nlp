import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from random import shuffle
import logging

# Import objects and functions customized for the abstractive summarization
from NPLM import NPLM_Summarizer
from Decoder import Greedy_Decoder
from data import Collection
from utils import load_glove_matrix, to_indices, collection_to_pairs, flatten


def evaluate(model, pairs, w2i, i2w):
    """
    Evaluate a model by generating
    """
    correct = 0

    for sequence, summary, continuation in pairs:
        sequence_i = Variable(torch.LongTensor(to_indices(sequence, w2i)))
        summary_i = Variable(torch.LongTensor(to_indices(summary, w2i)))
        scores = model.forward(sequence_i, summary_i)
        predict = scores.data.numpy().argmax(axis=1)[0]

        if predict == w2i[continuation]:
            correct += 1

    return correct, len(pairs), correct/len(pairs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Load data
    corpus = Collection("../opinosis/topics/", "../opinosis/summaries-gold/")
    w2i = corpus.dictionary.word2idx
    i2w = corpus.dictionary.idx2word

    context_size = 5

    train = collection_to_pairs(
        corpus.documents[:3], corpus.summaries[:3], w2i, context_size
    )
    logging.info("Loaded data.")
    embed = load_glove_matrix(w2i, "../glove.6B/glove.6B.300d.txt")
    logging.info("Initialized word embeddings with Glove.")

    # Sizes need for model and training
    embedding_dim = len(embed[0, :])
    vocab_dim = len(w2i)

    # Initalize the network
    model = NPLM_Summarizer(context_size, vocab_dim, embedding_dim, 300, "att", embed)
    opt = optim.Adam(params=model.parameters(), lr=0.001, weight_decay=1e-5)
    loss = nn.NLLLoss()
    decoder = Greedy_Decoder(w2i, i2w, context_size)
    logging.info("Initialized neural model and decoder.")

    for i in range(200):
        shuffle(train)

        for j, (sequence, summary, continuation) in enumerate(train):
            # Forward pass
            sequence_i = Variable(torch.LongTensor(to_indices(sequence, w2i)))
            summary_i = Variable(torch.LongTensor(to_indices(summary, w2i)))
            scores = model.forward(sequence_i, summary_i)

            # Calculate loss
            target = Variable(torch.LongTensor([w2i[continuation]]))
            output = loss(scores, target)

            # Backward pass
            opt.zero_grad()
            model.zero_grad()
            output.backward()
            opt.step()

            # Output accuracy
            if j % 1000 == 0:
                _, _, acc = evaluate(model, train, w2i, i2w)
                print("Epoch {}, iter {}, train acc={}".format(i, j, acc))

        # Output predicted summaries
        s = []
        s.append("\nEpoch {}\n---------------\n".format(i))
        for k, d in enumerate(corpus.documents[:3]):
            summary = decoder.decode(flatten(d.sentences), model)
            s.append("\n{}".format(d.name))
            s.append(" ".join(summary))
        open("summaries.txt", 'a').write("\n".join(s))
