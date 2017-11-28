import os
import torch
from collections import defaultdict


class Dictionary(object):

    def __init__(self):
        self.word2idx = defaultdict(lambda: len(self.word2idx))
        self.idx2word = dict()

    def add_word(self, word):
        self.word2idx[word]
        self.idx2word[self.word2index[word]] = word

    def add_text(self, text):
        for word in text:
            self.add_word(word)

    def to_unk(self):
        UNK = self.word2idx["<unk>"]
        self.word2idx = defaultdict(lambda: UNK, self.word2idx)


class Corpus(object):

    def __init__(self, path):
        self.train = Text(os.path.join(path, 'train.txt'))
        self.valid = Text(os.path.join(path, 'valid.txt'))
        self.test = Text(os.path.join(path, 'test.txt'))


class Text(object):

    def __init__(self, text, from_file=True):
        self.dictionary = Dictionary()
        if from_file:
            self.sentences, self.words = self.from_file(text)
        else:
            self.sentences, self.words = self.from_var(text)

    def from_file(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Tokenize file content
        with open(path, 'r') as f:
            sentences = [s.split() for s in f.readlines()]
        sentences, words = self.prepare(sentences)
        return sentences, words

    def from_var(self, sentences):
        sentences, words = self.prepare(sentences)
        return sentences, words

    def prepare(self, sentences):

        # Add start and end tags
        for i, s in enumerate(sentences):
            sentences[i] = ['<s>'] + s + ['</s>']

        # Add words to dictionary
        words = [word for s in sentences for word in s]
        self.dictionary.add_text(words)
        return sentences, words

class Collection(object):

    def __init__(self, path):
        self.path = path
        for f in os.listdir(path):

