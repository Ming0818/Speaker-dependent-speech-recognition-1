import argparse

from recognition import cross_validate, train, recognize

parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(dest="subparser_name")

cv_parser = subparsers.add_parser('cross_validate')
cross_validate.init_parser(cv_parser)

tr_parser = subparsers.add_parser('train')
train.init_parser(tr_parser)

re_parser = subparsers.add_parser('recognize')
recognize.init_parser(re_parser)

args = parser.parse_args()
if args.subparser_name == 'cross_validate':
    cross_validate.main(args)
elif args.subparser_name == 'train':
    train.main(args)
elif args.subparser_name == 'recognize':
    recognize.main(args)
