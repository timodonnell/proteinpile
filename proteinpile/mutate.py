import pandas
import uuid
import hashlib
import time
import optuna

import tqdm

from .common import add_pile_to_study

def add_args(parser):
    parser.add_argument(
        "action", choices=("add-sequences", "add-backbones", "hand-edit"))
    parser.add_argument("--num", type=int)
    parser.add_argument("--num-sequences-per-backbone", type=int)
    parser.add_argument("--designs", nargs="+")
    parser.add_argument("--optuna-optimize", default=False, action="store_true")


def action_add_backbones(args, pile, spec):
    df = pile.manifest
    print("Adding new backbones")

    study = optuna.create_study(directions=spec.directions())
    if args.optuna_optimize:
        add_pile_to_study(spec, pile, study)

    new_rows = []
    for i in range(args.num):
        new_row = {}
        new_row['method'] = 'rfdiffusion-proteinmpnn'
        trial = None
        while trial is None or not spec.are_params_feasible(trial.params):
            trial = study.ask(spec.get_distributions())
        new_row['params_dict'] = trial.params
        new_row['params_dict']['_time'] = time.time()
        print("Sampled params")
        print(new_row['params_dict'])
        new_row["provenance_dict"] = {
            "mutate_command": {
                "action": "add-backbones",
                "num": args.num,
            },
        }
        base_name = hashlib.sha1(str(time.time()).encode()).hexdigest()
        for j in range(args.num_sequences_per_backbone):
            new_row = pandas.Series(new_row, name=f"{base_name}.seq_{j}")
            new_rows.append(new_row)

    new_rows = pandas.DataFrame(new_rows)
    print("New designs")
    print(new_rows)

    print("Original manifest size", len(pile.manifest))
    pile.manifest = pandas.concat([pile.manifest, new_rows])
    print("New manifest size", len(pile.manifest))
    pile.save()


def action_hand_edit(args, pile, spec):
    print("Edit pile.manifest and call pile.save()")
    import ipdb
    ipdb.set_trace()


def action_add_sequences(args, pile, spec):
    df = pile.manifest
    print("Adding new sequences for exisisting backbones")
    assert args.designs is not None
    sub_df = df.loc[
        args.designs,
        ['method', 'params_dict', 'backbone_filename', 'provenance_dict']
    ].copy()
    print("Selected designs:")
    print(sub_df)
    assert args.num is not None

    new_rows = []
    existing_names = set(df.index)
    for original_name, row in sub_df.iterrows():
        for i in range(args.num):
            new_row = row.copy()
            new_row.provenance_dict = {}
            new_row.provenance_dict["mutate_command"] = {
                "action": "add-sequences",
                "original_name": original_name,
                "num": args.num,
            }
            name_base = original_name
            if original_name.split(".")[-1].startswith("seq_"):
                name_base = ".".join(original_name.split(".")[:-1])
            make_name = lambda j: f"{name_base}.seq_{j}"

            j = 0
            while make_name(j) in existing_names:
                j += 1
            name = make_name(j)
            new_row.name = name
            existing_names.add(name)
            new_rows.append(new_row)

    new_rows = pandas.DataFrame(new_rows)
    print("New designs")
    print(new_rows)

    print("Original manifest size", len(pile.manifest))
    pile.manifest = pandas.concat([pile.manifest, new_rows])
    print("New manifest size", len(pile.manifest))
    pile.save()


def handle_mutate(args, pile, spec):
    if args.action == "add-sequences":
        action_add_sequences(args, pile, spec)
    elif args.action == "add-backbones":
        action_add_backbones(args, pile, spec)
    elif args.action == "hand-edit":
        action_hand_edit(args, pile, spec)
    else:
        raise NotImplementedError()







