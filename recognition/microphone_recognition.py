import argparse
from itertools import groupby

import librosa
from utils import guess, get_train_test_data, record, trim


def train_and_guess(y, sample_rate, verbose):
    mfcc = librosa.feature.mfcc(y, sample_rate, n_mfcc=13)

    train_data, _ = get_train_test_data()
    _, best_guess = guess(mfcc.T, train_data, verbose)
    return best_guess


def multiple_mode(filename, args):
    repetitions_threshold = int(args.repetitions_threshold)

    y, sample_rate = librosa.load(filename)
    delta = librosa.feature.delta(y) * 10000
    l = [x < int(args.delta_threshold) for x in delta] + [False, True]

    offset = 0
    grouped_l = []
    buffered_value = True
    buffered_offset = 0
    buffered_repetitions = 0

    for value, g in groupby(l):
        num_repetitions = sum(1 for _ in g)

        if buffered_value == value and num_repetitions > 5:
            buffered_repetitions += num_repetitions
        elif (buffered_value != value and num_repetitions > 5) or \
                (buffered_value == value and num_repetitions < 5):

            interval = (buffered_value, buffered_repetitions, buffered_offset)
            if buffered_repetitions > repetitions_threshold:
                grouped_l.append(interval)
            buffered_value = value
            buffered_repetitions = num_repetitions
            buffered_offset = offset

        offset += num_repetitions

    cutting_points = []
    for t in grouped_l:
        cutting_points.append(t[2] + t[1] / 2)

    if args.verbose:
        print("  (< threshold, length, offset) where length > %d" % int(args.repetitions_threshold))
        print(grouped_l)
        #
        # plt.figure()
        # plt.subplot(4, 1, 1)
        # plt.plot(y)
        # plt.subplot(4, 1, 2)
        # plt.plot(delta)
        # plt.subplot(4, 1, 3)
        # plt.plot(l)
        # plt.subplot(4, 1, 4)
        # plt.plot(y)
        # for x in cutting_points:
        #     plt.plot((x, x), (-.5, .5), 'r-')
        #
        # plt.show()

    for i in range(len(cutting_points) - 1):
        start = cutting_points[i]
        stop = cutting_points[i + 1]
        y_interval = y[start:stop]
        filename = "../temp/cut_%d.wav" % i
        librosa.output.write_wav(filename, y_interval, sample_rate)

        y_interval, sample_rate = trim(filename)
        print(train_and_guess(y, sample_rate, args.verbose))


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
        y, sample_rate = record(0 if args.s else .5)
        librosa.output.write_wav(filename, y, sample_rate)

    if args.s:
        y, sample_rate = librosa.load(filename)
        print(train_and_guess(y, sample_rate, args.verbose))
    else:
        multiple_mode(filename, args)


if __name__ == '__main__':
    main()
