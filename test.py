import argparse

import Extractive.run_sumy as extr


def main():
    extr.main()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument()
    args = parser.parse_args()
    print(args)

    main()
