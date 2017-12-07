import argparse
import sumy

from sumy.parsers.html import HtmlParser
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words


LANGUAGE = "english"
SENTENCES_COUNT = 1


def main():
    url = "http://www.spiegel.de/international/europe/as-brexit-nears-harrassment-of-eu-citizens-in-uk-rises-a-1181845.html"
    parser = HtmlParser.from_url(url, Tokenizer(LANGUAGE))
    # or for plain text files
    # parser = PlaintextParser.from_file("document.txt", Tokenizer(LANGUAGE))
    stemmer = Stemmer(LANGUAGE)
    run_LSA(stemmer, parser.document)
    run_LexRank(stemmer, parser.document)
    run_TextRank(stemmer, parser.document)
    run_Luhn(stemmer, parser.document)


def run_LSA(stemmer, document):
    lsa = LsaSummarizer(stemmer)
    lsa.stop_words = get_stop_words(LANGUAGE)
    print("LSA")
    for sentence in lsa(document, SENTENCES_COUNT):
        print(sentence)


def run_LexRank(stemmer, document):
    lex = LexRankSummarizer(stemmer)
    lex.stop_words = get_stop_words(LANGUAGE)
    print("LexRank")
    for sentence in lex(document, SENTENCES_COUNT):
        print(sentence)


def run_TextRank(stemmer, document):
    text = TextRankSummarizer(stemmer)
    text.stop_words = get_stop_words(LANGUAGE)
    print("TextRank")
    for sentence in text(document, SENTENCES_COUNT):
        print(sentence)


def run_Luhn(stemmer, document):
    luhn = LuhnSummarizer(stemmer)
    luhn.stop_words = get_stop_words(LANGUAGE)
    print("Luhn")
    for sentence in luhn(document, SENTENCES_COUNT):
        print(sentence)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument()
    args = parser.parse_args()
    print(args)

    main()
