import os

import librosa

from recognition.base import record, guess, get_train_test_data


def main():
    audio = record()

    filename = "../temp/spoken.wav"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        f.write(audio.get_wav_data())
    f.close()

    y, sample_rate = librosa.load(filename)
    mfcc = librosa.feature.mfcc(y, sample_rate, n_mfcc=13)

    train_data, _ = get_train_test_data()
    _, best_guess = guess(mfcc.T, train_data)

    print(best_guess)


if __name__ == '__main__':
    main()
