import argparse
import os
from shutil import copyfile

from utils import record, remove_silence


def load_examples():
    for subdir, dirs, files in os.walk("../data/recordings/"):
        if subdir[-1] == "/":
            continue
        name = subdir.split("/")[-1]
        for filename in files:
            if 'wav' not in filename:
                continue
            src = '../data/recordings/%s/%s' % (name, filename)
            dst = '../recordings/%s/%s' % (name, filename)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            copyfile(src, dst)


def train(name):
    while True:
        recording_path = record()

        i = 0
        while os.path.exists("../recordings/%s/%s.wav" % (name, i)):
            i += 1
        filename = "../recordings/%s/%s.wav" % (name, i)
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        remove_silence(recording_path, filename)

        s = input("Press return to add another one, type 'q' to quit:")
        if s != "":
            break


def main():
    parser = argparse.ArgumentParser(description='Record audio.')
    parser.add_argument('-l', '--load-examples', dest='load_examples', action='store_true',
                        default=False, help='Load example training data')
    parser.add_argument('-n', '--name', dest="name", help='Word that will be spoken')
    args = parser.parse_args()

    if args.load_examples:
        load_examples()
    else:
        train(name=args.name)


if __name__ == '__main__':
    main()
