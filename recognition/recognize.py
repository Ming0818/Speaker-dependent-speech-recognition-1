from __future__ import print_function

from itertools import groupby

import librosa
import matplotlib.pyplot as plt

from recognition.utils import guess, record, save_audio, remove_silence


def train_and_guess(y, sample_rate, verbose):
    mfcc = librosa.feature.mfcc(y, sample_rate, n_mfcc=13)
    _, best_guess = guess(mfcc.T, verbose)
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

        plt.figure()
        fig = plt.subplot(4, 1, 1)
        fig.set_title('Audio waveform', fontsize=14)
        plt.plot(y)
        fig = plt.subplot(4, 1, 2)
        fig.set_title('Audio waveform delta', fontsize=14)
        plt.plot(delta)
        fig = plt.subplot(4, 1, 3)
        fig.set_title('Delta > threshold', fontsize=14)
        plt.plot(l)
        fig = plt.subplot(4, 1, 4)
        fig.set_title('Audio waveform with cutting points', fontsize=14)
        plt.plot(y)
        for x in cutting_points:
            plt.plot((x, x), (-.3, .3), 'r-')

        plt.gcf().set_tight_layout(True)
        plt.show()

    for i in range(len(cutting_points) - 1):
        start = int(cutting_points[i] - cutting_points[i] % 16)
        stop = int(cutting_points[i + 1] - cutting_points[i + 1] % 16)
        y_interval = y[start:stop]

        filename = "./temp/cut_%d.wav" % i
        save_audio(y_interval, sample_rate, filename)

        trimmed = filename.replace(".wav", ".trimmed.wav")
        remove_silence(filename, trimmed)

        y_interval, sample_rate = librosa.load(trimmed)
        print(train_and_guess(y_interval, sample_rate, args.verbose))


def init_parser(parser):
    parser.add_argument('-s', '--single', dest='s', action='store_const',
                        const=True, default=False, help='Single word mode')
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_const',
                        const=True, default=False, help='Show detailed info')
    parser.add_argument('-l', '--last', dest='use_last', const=True, action='store_const',
                        default=False, help='Use last mic recording instead of listening')
    parser.add_argument('-d', '--delta', dest='delta_threshold', default=5,
                        help='Delta threshold used for cutting')
    parser.add_argument('-r', '--repetitions', dest='repetitions_threshold', default=3000,
                        help='Repetitions threshold used for cutting')
    parser.add_argument('-f', '--filename', dest='filename', default=None,
                        help='Filename of the recording to use')


def main(args):
    filename = "./temp/buffer.wav"
    if args.filename is not None:
        filename = args.filename
        print("Using: %s" % filename)
    elif not args.use_last:
        filename = record()

    if args.s:
        y, sample_rate = librosa.load(filename)
        print(train_and_guess(y, sample_rate, args.verbose))
    else:
        multiple_mode(filename, args)
