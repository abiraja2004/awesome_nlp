import os
import nltk
from collections import defaultdict


class Dictionary(object):

    def __init__(self):
        self.word2idx = defaultdict(lambda: len(self.word2idx))
        self.idx2word = dict()

    def add_word(self, word):
        index = self.word2idx[word]
        self.idx2word[index] = word
        return index

    def add_text(self, text):
        for word in text:
            self.add_word(word)

    def to_unk(self):
        UNK = self.add_word("unk")
        self.word2idx = defaultdict(lambda: UNK, self.word2idx)


class Corpus(object):

    def __init__(self, path):
        self.train = Text(os.path.join(path, 'train.txt'))
        self.valid = Text(os.path.join(path, 'valid.txt'))
        self.test = Text(os.path.join(path, 'test.txt'))


class Text(object):

    def __init__(self, text, from_file=True):
        self.name = text.split('/')[-1]
        self.dictionary = Dictionary()
        if from_file:
            self.sentences, self.words = self.from_file(text)
        else:
            self.sentences, self.words = self.from_var(text)

    def from_file(self, path):
        assert os.path.exists(path)

        # Tokenize file content
        with open(path, 'r', errors='ignore') as f:
            sentences = nltk.sent_tokenize(f.read())
        sentences = [nltk.word_tokenize(s) for s in sentences]
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

    def __init__(self, documents_path, summaries_path):
        self.documents_path = documents_path
        self.summaries_path = summaries_path
        self.documents = []
        self.summaries = []
        self.dictionary = Dictionary()

        for f in os.listdir(documents_path):
            document = Text(os.path.join(documents_path, f))
            self.documents.append(document)
            self.dictionary.add_text(document.words)

        for folder in os.listdir(summaries_path):
            summaries = []
            for file in os.listdir(os.path.join(summaries_path, folder)):
                summary = Text(os.path.join(summaries_path, folder, file))
                summaries.append(summary)
            self.summaries.append(summaries)

        self.dictionary.to_unk()


if __name__ == "__main__":
    docs = Collection("../opinosis/topics/", "../opinosis/summaries-gold/")
    print(docs.documents[1].name)
    for s in docs.summaries[1]:
        print(s.name)
