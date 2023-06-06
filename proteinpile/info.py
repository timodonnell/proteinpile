import numpy
import tqdm
import json
import optuna
import pandas

from . import common


def add_args(parser):
    parser.add_argument("--verify", action="store_true", default=False)
    parser.add_argument("--out-pymol", metavar="SCRIPT.pml")
    parser.add_argument(
        "--selection-criteria",
        choices=("pareto", "interactive", "expression"),
        default="pareto")
    parser.add_argument("--expression")


def handle_info(args, pile, spec):
    print("Loaded manifest of shape", *pile.manifest.shape)
    for col in pile.manifest.columns:
        print(
            f"\t{col} {pile.manifest[col].isnull().sum()} null, "
            f"{pile.manifest[col].map(json.dumps).nunique()} non-null unique")

    if 'metrics_dict' in pile.manifest:
        pile.summarize_metrics()
        best_names = []
        best_df = None
        if args.selection_criteria == "pareto":
            study = optuna.create_study(directions=spec.directions())
            trial_num_to_name = common.add_pile_to_study(spec, pile, study)
            for (i, trial) in enumerate(study.best_trials):
                name = trial_num_to_name[trial.number]
                best_names.append(name)
        elif args.selection_criteria == "interactive":
            print("Define 'best_names' to be a list of the IDs of your selected trials.")
            print("You can also define best_df to be a dataframe of the selected designs.")
            print("pile.manifest is:")
            print(pile.manifest)
            df = pile.manifest.copy()
            df = pandas.concat(
                [df] + [
                    pandas.DataFrame.from_records(df[col].values, index=df.index)
                    for col in df.columns
                    if col.endswith("_dict")
                ],
                axis=1)
            import ipdb
            ipdb.set_trace()
        elif args.selection_criteria == "expression":
            best_names = pile.manifest.query(args.expression).index.tolist()
        else:
            raise ValueError(f"Unknown selection criteria {args.selection_criteria}")

        if best_df is not None:
            best_names = best_df.index.values
        best_df = pile.manifest.loc[best_names]
        for (i, (name, row)) in enumerate(best_df.iterrows()):
            print("*" * 40)
            print(f"BEST TRIAL {i + 1}/{len(study.best_trials)}")
            row = pile.manifest.loc[name]
            print(name)
            print(row)
            for col in ["af2_filename", "omegafold_filename"]:
                print("\t%40s : %s" % (col, row[col]))
            print("Seq: ", row.seq)
            for col in ["metrics_dict", "af2_dict", "omegafold_dict"]:
                if len(row[col]) > 0:
                    print("%s:" % col)
                    for k, v in row[col].items():
                        if type(v) != float:
                            continue
                        min_value = best_df[col].map(lambda d: d.get(k)).dropna().min()
                        max_value = best_df[col].map(lambda d: d.get(k)).dropna().max()
                        annotation = " [best range %0.5f - %0.5f]" % (min_value, max_value)
                        if v == min_value:
                            annotation += " [lowest]"
                        if v == max_value:
                            annotation += " [highest]"
                        print("\t%40s : %0.5f" % (k, v) + annotation)

        if args.out_pymol:
            with open(args.out_pymol, "w") as fd:
                designs = pile.get_designs(spec, best_df.index.tolist())
                lines = spec.pymol_lines(designs)
                for line in lines:
                    fd.write(line)
                    fd.write("\n")
            print("Wrote pymol script to", args.out_pymol)

    if args.verify:
        df = pile.manifest
        for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
            if row.seq is not None:
                problem = common.get_problem(spec, row.params_dict)
                problem.check_solution_sequence_is_valid(row.seq)
                if row.backbone_filename is not None:
                    print(row.backbone_filename)
                    backbone = pile.load_pdb(row.backbone_filename)
                    numpy.testing.assert_equal(len(backbone.select("chain A").ca), len(row.seq))
                if row.af2_filename is not None:
                    pred = pile.load_pdb(row.af2_filename)
                    numpy.testing.assert_equal(pred.ca.getSequence(), row.seq)
                if row.omegafold_filename is not None:
                    pred = pile.load_pdb(row.omegafold_filename)
                    numpy.testing.assert_equal(pred.ca.getSequence(), row.seq)



