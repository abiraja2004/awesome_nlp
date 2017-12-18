import os
import nltk
import logging


class Dictionary(object):
    """
    Object that creates and keeps word2idx and idx2word dicts.
    Do not forget to call to_unk when the word2idx dict is filled.
    """
    def __init__(self):
        self.word2idx = defaultdict(lambda: len(self.word2idx))
        self.idx2word = dict()
        self.add_word("<s>")
        self.add_word("</s>")

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
        self.end_tags = end_tags
        if from_file:
            self.text = self.from_file(text)
            self.name = text.split('/')[-1]
        else:
            self.text = self.from_var(text)

    def from_file(self, path):
        """
        Read a text from a filepath.
        """
        assert os.path.exists(path)

        # Tokenize file content
        with open(path, 'r', errors='ignore') as f:
            sentences = nltk.sent_tokenize(f.read())
        sentences = [nltk.word_tokenize(s) for s in sentences]
        text = self.prepare(sentences)
        sentences = []
        return text

    def from_var(self, sentences):
        """
        Read a text from a string.
        """
        sentences = nltk.sent_tokenize(sentences)
        sentences = [nltk.word_tokenize(s) for s in sentences]
        text = self.prepare(sentences)
        sentences = []
        return text

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
        sentences = []
        return words


class Gigaword_Collection(object):
    """
    Collects documents and corresponding summaries.
    """
    def __init__(self, documents_file, summaries_file, nr_docs):
        if nr_docs == 0:
            nr_docs = -1
        self.documents_path = documents_file
        self.summaries_path = summaries_file
        self.documents = open(self.documents_path, 'r').readlines()[:nr_docs]
        self.summaries = open(self.summaries_path, 'r').readlines()[:nr_docs]
        self.dictionary = Dictionary()
        self.pairs = []

        # Extract all text and fill dicts
        total = len(self.documents)
        for i in range(total):
            if i % 100000 == 0:
                logging.debug("Loading documents, {} / {}.".format(i, total))
            self.dictionary.add_text(set(self.prepare(self.documents[i])))

        for i in range(total):
            if i % 100000 == 0:
                logging.debug("Loading summaries, {} / {}.".format(i, total))
            self.dictionary.add_text(set(self.prepare(self.summaries[i])))

        self.dictionary.to_unk()
        logging.info("Initialized corpus.")

    def collection_to_pairs(self):
        """
        Create training pairs for the entire collection.
        """
        n = len(self.documents)
        for i, document in enumerate(self.documents):
            if i % 100000 == 0:
                logging.debug("Preparing pairs for doc {}/{}.".format(i, n))
            document = self.to_indices(self.prepare(document))
            summary = self.to_indices(self.prepare(self.summaries[i]))
            self.pairs.append((document, summary))
        logging.info("Initialized training pairs.")
        return self.pairs

    def to_indices(self, sequence):
        """
        Represent a history of words as a list of indices.
        """
        return [self.dictionary.word2idx[w] for w in sequence]

    def prepare(self, sentence):
        """
        Add start and end tags (end tags only if preferred).
        Add words to the dictionary.
        """
        # Add start and end tags
        return ['<s>'] + nltk.word_tokenize(sentence) + ['</s>']
