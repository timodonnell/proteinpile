import argparse
import shlex
import optuna
import tqdm
import prody
import proteopt.client

from . import specification_defaults

def add_pile_to_study(spec, pile, study):
    distributions = spec.get_distributions()
    df = pile.manifest.loc[~pile.manifest.metrics_dict.isnull()]
    print("Loading data into optuna study")
    trial_num_to_name = {}
    for (i, design) in enumerate(tqdm.tqdm(pile.get_designs(spec, df.index.values))):
        trial = optuna.trial.create_trial(
            params=dict(
                (k, v) for (k, v) in design.params_dict.items() if not k.startswith("_")),
            distributions=distributions,
            values=spec.loss(design)
        )
        trial.number = i
        study.add_trial(trial)
        trial_num_to_name[i] = design.name
    return trial_num_to_name

PDB_CACHE = {}
def load_pdb(filename):
    if filename not in PDB_CACHE:
        PDB_CACHE[filename] = prody.parsePDB(filename)
    return PDB_CACHE[filename]


def get_specification_variables(filename):
    with open(filename) as f:
        vars = {}
        code = compile(f.read(), filename, 'exec')
        exec(code, vars, vars)
        return vars


def get_spec(filename, args=""):
    spec_vars = get_specification_variables(filename)
    spec_parser = argparse.ArgumentParser()
    spec_vars['Specification'].add_args(spec_parser)
    spec_args = spec_parser.parse_args(shlex.split(args))
    spec = spec_vars['Specification'](spec_args)

    if spec.required_structure_predictors is None:
        print("Using default required_structure_predictors() implementation")
        spec.required_structure_predictors = \
            specification_defaults.required_structure_predictors
    if spec.get_metrics is None:
        print("Using default get_metrics() implementation")
        spec.get_metrics = specification_defaults.get_metrics
    if spec.loss is None:
        print("Using default loss() implementation")
        spec.loss = specification_defaults.loss
    if spec.directions is None:
        print("Using default directions() implementation")
        spec.directions = specification_defaults.directions
    return spec


def get_runner(client, cls, local_conf={}, **kwargs):
    if client is not None:
        runner = client.remote_model(cls, **kwargs)
    else:
        runner = cls(**local_conf, **kwargs)
    return runner


def get_client(args):
    if args.endpoints_file:
        endpoints = []
        for f in args.endpoints_file:
            endpoints.extend(x.strip() + "/tool" for x in open(f).readlines() if x.strip())
        print("Using endpoints [%d]" % len(endpoints), endpoints)
        client = proteopt.client.Client(
            endpoints=endpoints,
            extra_parallelism_factor=args.client_extra_parallelism_factor)
    else:
        print("No endpoints, will run locally")
        client = None
    return client


PROBLEM_CACHE = {}
def get_problem(spec, params_dict):
    if params_dict not in PROBLEM_CACHE:
        PROBLEM_CACHE[params_dict] = spec.get_problem(params_dict)
    return PROBLEM_CACHE[params_dict]