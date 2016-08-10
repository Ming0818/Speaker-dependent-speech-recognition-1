import os
from math import inf

import librosa
import speech_recognition as sr
from dtw import dtw
from numpy.dual import norm


def get_train_test_data(num_tests=0):
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


def guess(target_mfcc, train_data, verbose=False):
    d_min, name_min = inf, ""
    for train_name, train_mfcc in train_data:
        d, _, _, _ = dtw(target_mfcc, train_mfcc, dist=lambda x, y: norm(x - y, ord=1))

        if verbose:
            print("  %s %f" % (train_name, d))

        if d < d_min:
            d_min = d
            name_min = train_name
    return d_min, name_min


def record(non_speaking_duration=0):
    # obtain audio from the microphone
    r = sr.Recognizer()
    r.non_speaking_duration = non_speaking_duration

    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)  # listen for 1 second to calibrate the
        # ################################### energy threshold for ambient noise levels

        print("Listening...")
        audio = r.listen(source)
    return audio
