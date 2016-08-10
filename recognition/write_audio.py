import argparse
import os

from recognition.base import record


def main():
    parser = argparse.ArgumentParser(description='Record audio.')
    parser.add_argument('name', nargs=1, help='word that will be spoken')
    args = parser.parse_args()

    name = args.name[0]
    audio = record()

    i = 0
    while os.path.exists("../recordings/%s/%s.wav" % (name, i)):
        i += 1
    filename = "../recordings/%s/%s.wav" % (name, i)

    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())


if __name__ == '__main__':
    main()
