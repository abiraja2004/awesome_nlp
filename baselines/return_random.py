import random


def gen_sum(document, words=12):
    wordlist = document.split()
    return " ".join(random.sample(wordlist, words))
