import argparse
import os

from utils import record, remove_silence


def main():
    parser = argparse.ArgumentParser(description='Record audio.')
    parser.add_argument('name', nargs=1, help='Word that will be spoken')
    args = parser.parse_args()

    name = args.name[0]

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


if __name__ == '__main__':
    main()
