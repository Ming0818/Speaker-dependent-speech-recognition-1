import argparse

from utils import get_train_test_data, guess


def cross_validate(num_tests, verbose):
    train_data, test_data = get_train_test_data(num_tests)
    score = 0

    for test_name, test_mfcc in test_data:
        d_min, name_min = guess(test_mfcc, train_data, verbose)
        correct = test_name == name_min

        if correct:
            score += 1.0

        if verbose:
            print("%s - Target: %s - Guess: %s" % ("Ok" if correct else "Failed", test_name, name_min))
        else:
            if correct:
                print('.', end='', flush=True)
            else:
                print('F', end='', flush=True)
    print()
    return score / len(test_data)


def main():
    parser = argparse.ArgumentParser(description='Cross validate using training set.')
    parser.add_argument('--verbose', '-v', dest='verbose', action='store_const',
                        const=True, default=False, help='Show detailed info')
    parser.add_argument('num_tests', nargs=1, help='Number of recordings per word '
                                                   'to reserve for tests')
    args = parser.parse_args()
    num_tests = int(args.num_tests[0])
    score = cross_validate(num_tests, args.verbose) * 100

    print('Recognition rate %.2f%%' % score)


if __name__ == '__main__':
    main()
