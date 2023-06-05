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
    spec_vars = common.get_specification_variables(args.specification)

    spec_parser = argparse.ArgumentParser()
    spec_vars['Specification'].add_args(spec_parser)
    spec_args = spec_parser.parse_args(shlex.split(args.specification_args))
    spec = spec_vars['Specification'](spec_args)

    if spec.required_structure_predictors is None:
        print("Using default required_structure_predictors() implementation")
        spec.required_structure_predictors = specification_defaults.required_structure_predictors
    if spec.get_metrics is None:
        print("Using default get_metrics() implementation")
        spec.get_metrics = specification_defaults.get_metrics
    if spec.loss is None:
        print("Using default loss() implementation")
        spec.loss = specification_defaults.loss
    if spec.directions is None:
        print("Using default directions() implementation")
        spec.directions = specification_defaults.directions

    if args.subcommand == "evaluate":
        evaluate.handle_evaluate(args, pile, spec)
    elif args.subcommand == "mutate":
        mutate.handle_mutate(args, pile, spec)
    elif args.subcommand == "info":
        info.handle_info(args, pile, spec)

