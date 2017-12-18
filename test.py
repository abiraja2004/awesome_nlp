import argparse
import os

import baselines.run_sumy as extr
import baselines.return_first as first
import baselines.return_random as rd
import gensim


SUMM_DIR = "generated_summaries"


def generate_summaries(input_file, n):
    with open(input_file, 'r') as document:
        for line in document:
            # Generate extractive sentences
            summ = extr.gen_sum(line, n, "SumBasic")
            with open(os.path.join(SUMM_DIR, "SumBasic_generated_summaries.txt"), 'w') as f:
                f.write(str(summ) + "\n")


def naive_baseline(input_file, n):
    print("Naive: {}".format(n))
    with open(input_file, 'r') as document:
        for line in document:
            with open(os.path.join(SUMM_DIR, "naive_generated_summaries.txt"), "a") as f:
                f.write(first.gen_sum(line, n) + "\n")


def random_baseline(input_file, n):
    print("Random: {}".format(n))
    with open(input_file, 'r') as document:
        for line in document:
            with open(os.path.join(SUMM_DIR, "random_generated_summaries.txt"), "a") as f:
                f.write(rd.gen_sum(line, n) + "\n")


def pke_baseline():
    print("PKE: {}".format(n))
    with open(input_file, 'r') as document:
        for line in document:
            with open(os.path.join(SUMM_DIR, "pke_generated_summaries.txt"), "a") as f:
                f.write(pk.gen_sum(line, n) + "\n")


def main(input_file, n):
    # Create the SUMM_DIR
    if not os.path.exists(SUMM_DIR):
        os.makedirs(SUMM_DIR)
    naive_baseline(input_file, n)
    random_baseline(input_file, n)
    print("Generated summaries in: {}".format(os.path.realpath(SUMM_DIR)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="inputs.txt")
    parser.add_argument("--words", default="8", help="how many words the \
    summary should maximally contain", type=int)
    args = parser.parse_args()
    print(args)

    main(args.input, args.words)
