import sys
import os
import pke

SUMM_DIR = "generated_summaries"


def gen_sum(document, words=12):
    with open("tmp.txt", 'w') as f:
        f.write(document)
    extractor = pke.TopicRank("tmp.txt")

    extractor.read_document(format="raw", stemmer=None)
    extractor.candidate_selection()
    extractor.candidate_weighting()
    keyphrases = extractor.get_n_best(n=10)
    return str(keyphrases[0][0])


def main(input_file, n):
    with open(input_file, 'r') as document:
        for i, line in enumerate(document):
            with open(os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/" + SUMM_DIR, "pke_generated_summaries.txt"), "a") as f:
                try:
                    summ = gen_sum(line, n)
                except Exception as e:
                    print(line)
                    print("Summ: {}".format(summ))
                    print("On summ: {}".format(i))
                    print(e)

                f.write(summ + "\n")


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[int(2)])
