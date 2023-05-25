import sys
import argparse
import shlex
import os

import tqdm
import pandas
import prody
import proteopt


from .pile import Pile
from . import common, evaluate, mutate, verify

parser = argparse.ArgumentParser()
parser.add_argument("--manifest", default="manifest.csv")
parser.add_argument("--intermediates-dir")

parser.add_argument("--specification", default="specification.py")
parser.add_argument("--specification-args", default="")

subparsers = parser.add_subparsers(required=True, dest="subcommand")

subparser = subparsers.add_parser('new')

subparser = subparsers.add_parser('evaluate')
evaluate.add_args(subparser)

subparser = subparsers.add_parser('mutate')
mutate.add_args(subparser)

subparser = subparsers.add_parser('verify')
verify.add_args(subparser)


def run(argv=sys.argv[1:]):
    args = parser.parse_args()
    print(args)

    if args.subcommand == "new":
        pile = Pile.blank(args.manifest)
        pile.save()
        return

    pile = Pile(args.manifest, args.intermediates_dir)
    spec_vars = common.get_specification_variables(args.specification)

    spec_parser = argparse.ArgumentParser()
    spec_vars['Specification'].add_args(spec_parser)
    spec_args = spec_parser.parse_args(shlex.split(args.specification_args))
    spec = spec_vars['Specification'](spec_args)

    if args.subcommand == "evaluate":
        evaluate.handle_evaluate(args, pile, spec)
    elif args.subcommand == "mutate":
        mutate.handle_mutate(args, pile, spec)
    elif args.subcommand == "verify":
        verify.handle_verify(args, pile, spec)

