import argparse
import os

import Extractive.run_sumy as extr


SUMM_DIR = "generated_summaries"


def generate_summaries(input_file):
    # Create the SUMM_DIR
    if not os.path.exists(SUMM_DIR):
        os.makedirs(SUMM_DIR)
    with open(input_file, 'r') as document:
        for line in document:
            # Generate extractive sentences
            summ = extr.gen_sum(line, "LSA")
            with open(os.path.join(SUMM_DIR, "LSA_generated_summaries.txt"), 'w') as f:
                f.write(str(summ[0]) + "\n")
            summ = extr.gen_sum(line, "LexRank")
            with open(os.path.join(SUMM_DIR, "LexRank_generated_summaries.txt"), 'w') as f:
                f.write(str(summ[0]) + "\n")
            summ = extr.gen_sum(line, "TextRank")
            with open(os.path.join(SUMM_DIR, "TextRank_generated_summaries.txt"), 'w') as f:
                f.write(str(summ[0]) + "\n")
            summ = extr.gen_sum(line, "Luhn")
            with open(os.path.join(SUMM_DIR, "Luhn_generated_summaries.txt"), 'w') as f:
                f.write(str(summ[0]) + "\n")


def main(input_file):
    # Returns a list of filenames which contain generated sentences
    gen_files = generate_summaries(input_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="inputs.txt")
    args = parser.parse_args()
    print(args)

    main(args.input)
