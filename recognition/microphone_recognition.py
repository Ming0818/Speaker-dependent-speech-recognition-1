import argparse
import os
from itertools import groupby

import librosa
import matplotlib.pyplot as plt
from utils import guess, get_train_test_data, record


def read_and_guess(filename, verbose):
    y, sample_rate = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y, sample_rate, n_mfcc=13)

    train_data, _ = get_train_test_data()
    _, best_guess = guess(mfcc.T, train_data, verbose)
    return best_guess


def multiple_mode(filename, args):
    y, sample_rate = librosa.load(filename)
    delta = librosa.feature.delta(y) * 10000
    l = [x < int(args.delta_threshold) for x in delta]

    offset = 0
    grouped_l = []
    for value, g in groupby(l):
        num_repetitions = sum(1 for _ in g)
        if num_repetitions > int(args.repetitions_threshold):
            grouped_l.append((value, num_repetitions, offset))
        offset += num_repetitions

    cutting_points = []
    for t in grouped_l:
        cutting_points.append(t[2] + t[1] / 2)

    if args.verbose:
        print("  (< threshold, length, offset) where length > %d" % int(args.repetitions_threshold))
        print(grouped_l)

        plt.figure()
        plt.subplot(4, 1, 1)
        plt.plot(y)
        plt.subplot(4, 1, 2)
        plt.plot(delta)
        plt.subplot(4, 1, 3)
        plt.plot(l)
        plt.subplot(4, 1, 4)
        plt.plot(y)
        for x in cutting_points:
            plt.plot((x, x), (-.5, .5), 'r-')

        plt.show()

    for i in range(len(cutting_points) - 1):
        start = cutting_points[i]
        stop = cutting_points[i + 1]
        audio = y[start:stop]
        filename = "../temp/cut_%d.wav" % i
        librosa.output.write_wav(filename, audio, sample_rate)
        print(read_and_guess(filename, args.verbose))


def main():
    parser = argparse.ArgumentParser(description='Cross validate.')
    parser.add_argument('-s', '--single', dest='s', action='store_const',
                        const=True, default=False, help='Single word mode')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_const',
                        const=True, default=False, help='Show detailed info')
    parser.add_argument('-l', '--last', dest='use_last', const=True, action='store_const',
                        default=False, help='Use last recording instead of listening')
    parser.add_argument('-d', '--delta', dest='delta_threshold', default=4,
                        help='Delta threshold used for cutting')
    parser.add_argument('-r', '--repetitions', dest='repetitions_threshold', default=1800,
                        help='Repetitions threshold used for cutting')
    args = parser.parse_args()

    filename = "../temp/spoken.wav"
    if not args.use_last:
        audio = record(0 if args.s else .5)
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, "wb") as f:
            f.write(audio.get_wav_data())
        f.close()

    if args.s:
        print(read_and_guess(filename, args.verbose))
    else:
        multiple_mode(filename, args)


if __name__ == '__main__':
    main()
