import argparse
import os

import librosa
from utils import record


def main():
    parser = argparse.ArgumentParser(description='Record audio.')
    parser.add_argument('name', nargs=1, help='Word that will be spoken')
    args = parser.parse_args()

    name = args.name[0]

    while True:
        y, sample_rate = record()

        i = 0
        while os.path.exists("../recordings/%s/%s.wav" % (name, i)):
            i += 1
        filename = "../recordings/%s/%s.wav" % (name, i)

        librosa.output.write_wav(filename, y, sample_rate)

        s = input("Press return to add another one, type 'q' to quit:")
        if s != "":
            break


if __name__ == '__main__':
    main()
