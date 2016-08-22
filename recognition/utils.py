import os
import subprocess
from itertools import groupby
from math import inf

import librosa
import matplotlib.pyplot as plt
import speech_recognition as sr
from dtw import dtw
from numpy.dual import norm

test_data = train_data = None


def get_train_test_data(num_tests=0):
    global train_data, test_data
    if train_data is not None:
        return train_data, test_data

    train_data = []
    test_data = []

    for subdir, dirs, files in os.walk("../recordings/"):
        if subdir[-1] == "/":
            continue

        name = subdir.split("/")[-1]

        curr_tests = 0
        for filename in files:
            if 'wav' not in filename:
                continue
            y, sample_rate = librosa.load('../recordings/%s/%s' % (name, filename))
            mfcc = librosa.feature.mfcc(y, sample_rate, n_mfcc=13)
            if curr_tests < num_tests:
                test_data.append((name, mfcc.T))
                curr_tests += 1
            else:
                train_data.append((name, mfcc.T))
    return train_data, test_data


def guess(target_mfcc, verbose=False):
    global train_data
    if train_data is None:
        get_train_test_data()

    d_min, name_min = inf, ""
    results = []
    for train_name, train_mfcc in train_data:
        d, _, _, _ = dtw(target_mfcc, train_mfcc, dist=lambda x, y: norm(x - y, ord=1))

        if verbose:
            results.append((train_name, d))

        if d < d_min:
            d_min = d
            name_min = train_name

    if verbose:
        plt.figure()
        plt.title('Difference between current word and training example', fontsize=20)
        plt.xlabel('Training example #', fontsize=16)
        plt.ylabel('Error', fontsize=16)
        for key, group in groupby(results, key=lambda x: x[0]):
            y = [x[1] for x in group]
            print(y, key)
            plt.plot(y, 'o-', label=key)
        plt.legend()
        plt.show()

    return d_min, name_min


def record():
    with sr.Microphone(sample_rate=22050) as source:
        filename = process_source(source)
        return filename


def remove_silence(input_path, output_path, block=True):
    sox_args = ['sox', input_path, '-c', '1', output_path, 'silence', '1', '0.1', '0.1%', '-1', '0.1', '0.1%']
    process_handle = subprocess.Popen(sox_args, stderr=subprocess.PIPE)
    if block:
        process_handle.communicate()
    return output_path


def process_source(source):
    non_speaking_duration = 0.5
    r = sr.Recognizer()
    r.non_speaking_duration = non_speaking_duration
    r.pause_threshold += non_speaking_duration
    r.adjust_for_ambient_noise(source)

    print("Listening...")
    audio = r.listen(source)
    print("Done.")

    filename = "../temp/buffer.wav"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())
    f.close()

    return filename


def save_audio(y, sample_rate, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    librosa.output.write_wav(filename, y, sample_rate)
