import argparse

from recognition.base import get_train_test_data, guess


def cross_validate(num_tests, verbose):
    train_data, test_data = get_train_test_data(num_tests)
    score = 0

    for test_name, test_mfcc in test_data:
        if verbose:
            print("\n" + test_name + "\n")

        d_min, name_min = guess(test_mfcc, train_data, verbose)

        correct = test_name == name_min
        if correct:
            score += 1.0

        if not verbose:
            if correct:
                print('.', end='', flush=True)
            else:
                print('F', end='', flush=True)

    return score / len(test_data)


def main():
    parser = argparse.ArgumentParser(description='Cross validate.')
    parser.add_argument('--verbose', dest='verbose', action='store_const',
                        const=True, default=False,
                        help='show detailed info')
    parser.add_argument('num_tests', nargs=1, help='number of recordings per word '
                                                   'to reserve for tests')
    args = parser.parse_args()

    print(
        '\n'
        'Recognition rate {}%'.format(
            100. * cross_validate(
                int(args.num_tests[0]),
                args.verbose
            )
        )
    )


if __name__ == '__main__':
    main()
