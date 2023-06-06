import sys
import argparse
import shlex
import logging

import tqdm
import pandas
import prody
import proteopt


from .pile import Pile
from . import common, evaluate, mutate, info, specification_defaults

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="manifest.csv")
parser.add_argument("--intermediates-dir", default="intermediates")

parser.add_argument("--specification", default="specification.py")
parser.add_argument("--specification-args", default="")
parser.add_argument("--quiet", action="store_true")

subparsers = parser.add_subparsers(required=True, dest="subcommand")

subparser = subparsers.add_parser('new')

subparser = subparsers.add_parser('evaluate')
evaluate.add_args(subparser)

subparser = subparsers.add_parser('mutate')
mutate.add_args(subparser)

subparser = subparsers.add_parser('info')
info.add_args(subparser)


def run(argv=sys.argv[1:]):
    args = parser.parse_args(argv)
    print(args)

    if args.quiet:
        prody.confProDy(verbosity='error')
        logging.getLogger("proteopt").setLevel(logging.ERROR)
        logging.getLogger(".prody").setLevel(logging.ERROR)
        logging.getLogger().setLevel(logging.ERROR)

    if args.subcommand == "new":
        pile = Pile.blank(args.manifest)
        pile.save()
        return

    pile = Pile(args.manifest, args.intermediates_dir)
    spec = common.get_spec(args.specification, args.specification_args)

    if args.subcommand == "evaluate":
        evaluate.handle_evaluate(args, pile, spec)
    elif args.subcommand == "mutate":
        mutate.handle_mutate(args, pile, spec)
    elif args.subcommand == "info":
        info.handle_info(args, pile, spec)

