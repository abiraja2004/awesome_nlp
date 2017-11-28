import os
import nltk
import logging
from torch import LongTensor as LT
from collections import defaultdict


class Dictionary(object):
    """Object that creates and keeps word2idx and idx2word dicts.
    Do not forget to call to_unk when the word2idx dict is filled."""
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
    def __init__(self, text, from_file=True, end_tags=False):
        self.dictionary = Dictionary()
        self.end_tags = end_tags
        if from_file:
            self.sentences, self.words, self.text = self.from_file(text)
            self.name = text.split('/')[-1]
        else:
            self.sentences, self.words, self.text = self.from_var(text)

    def from_file(self, path):
        """
        Read a text from a filepath.
        """
        assert os.path.exists(path)

        # Tokenize file content
        with open(path, 'r', errors='ignore') as f:
            sentences = nltk.sent_tokenize(f.read())
        sentences = [nltk.word_tokenize(s) for s in sentences]
        sentences, words = self.prepare(sentences)
        text = [w for s in sentences for w in s]
        return sentences, words, text

    def from_var(self, sentences):
        """
        Read a text from a string.
        """
        sentences = nltk.sent_tokenize(sentences)
        sentences = [nltk.word_tokenize(s) for s in sentences]
        sentences, words = self.prepare(sentences)
        text = [w for s in sentences for w in s]
        return sentences, words, text

    def prepare(self, sentences):
        """
        Add start and end tags (end tags only if preferred).
        Add words to the dictionary.
        """
        # Add start and end tags
        for i, s in enumerate(sentences):
            sentences[i] = ['<s>'] + s
            if self.end_tags:
                sentences[i] = sentences[i] + ['</s>']

        # Add words to dictionary
        words = [word for s in sentences for word in s]
        self.dictionary.add_text(words)
        return sentences, words


class Opinosis_Collection(object):
    """Collects documents and corresponding summaries."""
    def __init__(self, documents_path, summaries_path):
        self.documents_path = documents_path
        self.summaries_path = summaries_path
        self.documents = []
        self.summaries = []
        self.dictionary = Dictionary()

        # Extract all text and fill dicts
        for f in os.listdir(documents_path):
            document = Text(os.path.join(documents_path, f), True)
            self.documents.append(document)
            self.dictionary.add_text(document.words)

        for folder in os.listdir(summaries_path):
            summaries = []
            for file in os.listdir(os.path.join(summaries_path, folder)):
                summary = Text(os.path.join(summaries_path, folder, file), True)
                summaries.append(summary)
            self.summaries.append(summaries)

        self.dictionary.to_unk()

    def collection_to_pairs(self, context):
        self.pairs = []
        for i, document in enumerate(self.documents):
            p = self.to_pairs(self.documents[i], self.summaries[i], context)
            self.pairs.extend(p)
        return self.pairs

    def to_pairs(self, document, summaries, size):
        document.name
        sequence = self.to_indices([w for w in document.text])
        pairs = []
        for summary in summaries:
            summary = [w for w in summary.text]
            summary = self.to_indices(['<s>'] * (size - 1) + summary)
            for i in range(size, len(summary)):
                pairs.append((sequence, summary[i-size:i], LT([summary[i]])))
        return pairs

    def to_indices(self, sequence):
        """
        Represent a history of words as a list of indices.
        """
        return LT([self.dictionary.word2idx[w] for w in sequence])


class Gigaword_Collection(object):
    """Collects documents and corresponding summaries."""
    def __init__(self, documents_file, summaries_file, nr_docs):
        self.documents_path = documents_file
        self.summaries_path = summaries_file
        self.documents = open(self.documents_path, 'r').readlines()[:nr_docs]
        self.summaries = open(self.summaries_path, 'r').readlines()[:nr_docs]
        self.dictionary = Dictionary()

        # Extract all text and fill dicts
        total = len(self.documents)
        for i, document in enumerate(self.documents):
            logging.debug("Loading documents, {} / {}.".format(i, total))
            document = Text(document, False, False)
            self.documents[i] = document
            self.dictionary.add_text(document.words)

        for i, summary in enumerate(self.summaries):
            logging.debug("Loading summaries, {} / {}.".format(i, total))
            summary = Text(summary, False, False)
            self.summaries[i] = summary
            self.dictionary.add_text(summary.words)

        self.dictionary.to_unk()
        logging.info("Initialized corpus.")

    def collection_to_pairs(self, context):
        """
        Create training pairs for the entire collection.
        """
        self.pairs = []
        docs = []
        total = len(self.documents)
        for i, document in enumerate(self.documents):
            logging.debug("Preparing pairs for doc {} / {}.".format(i, total))
            indices = self.to_indices(document.text)
            docs.append(indices)
            p = self.to_pairs(i, len(indices), self.summaries[i], context)
            self.pairs.extend(p)
        logging.info("Initialized training pairs.")
        return docs, self.pairs

    def to_pairs(self, doc_id, doc_length, summary, size):
        """
        Create training pairs from a document.
        Training pairs consist of (document id of artickle, length of original
        article, context window of summary, correction continuation)
        """
        summary = self.to_indices(['<s>'] * (size - 1) + summary.text)
        pairs = []
        for i in range(size, len(summary)):
            pairs.append((doc_id, doc_length, summary[i-size:i], [summary[i]]))
        return pairs

    def to_indices(self, sequence):
        """
        Represent a history of words as a list of indices.
        """
        return [self.dictionary.word2idx[w] for w in sequence]
