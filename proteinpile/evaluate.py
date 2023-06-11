import os
import hashlib
import time
import logging

import numpy

import proteopt.client
import proteopt.proteinmpnn
import proteopt.alphafold
import proteopt.omegafold
import proteopt.rfdiffusion_motif


import prody

import tqdm

from . import common

def add_args(parser):
    parser.add_argument("--chunksize", type=int, default=1000)
    parser.add_argument("--endpoints-file", nargs="+")
    parser.add_argument("--client-extra-parallelism-factor", type=int, default=1)
    parser.add_argument("--items-per-request", type=int, default=1)
    parser.add_argument("--alphafold-weights-dir", default="/data/static/alphafold-params/")
    parser.add_argument("--omegafold-weights-dir", default="/data/static/omegafold_ckpt/")
    parser.add_argument("--rfdiffusion-weights-dir", default="/data/static/rfdiffusion-params")
    parser.add_argument("--recompute-metrics", action="store_true")

def handle_evaluate(args, pile, spec):
    client = common.get_client(args)

    if args.recompute_metrics:
        print("Will recompute all metrics")
        pile.manifest["metrics_dict"] = None

    # Now run designs and process by chunks
    df = pile.manifest
    while df[pile.COLUMNS].isnull().any().any():
        pile.summarize_metrics()
        print("Rows missing any col:", df[pile.COLUMNS].isnull().any(axis=1).sum())
        missing_counts = df[pile.COLUMNS].isnull().sum(1).value_counts()
        print("Number of rows missing indicated number of columns:")
        print(missing_counts.sort_index())
        min_missing = missing_counts[missing_counts.index > 0].index.to_series().min()
        print(f"Next chunk will process rows missing {min_missing} computed columns")
        sub_df = df.loc[
            df[pile.COLUMNS].isnull().sum(1) == min_missing
        ]
        chunk_indices = sample_chunk_grouped_by_params_or_design_backbone_filename(
            sub_df, args.chunksize)
        print("Sampled chunk of length", len(chunk_indices))
        process_chunk_to_completion(
            args,
            client,
            spec,
            pile,
            chunk_indices)

    print("Done.")


def process_chunk_to_completion(
        args, client, spec, pile, chunk_indices):
    chunk_indices = sorted(chunk_indices)
    chunk_df = pile.manifest.loc[chunk_indices]

    # RFDiffusion
    needs_backbone = chunk_df.loc[chunk_df.backbone_filename.isnull()]
    if len(needs_backbone) > 0:
        print("Running RFDiffusion")
        run_rfdiffusion_chunk(args, client, spec, pile, needs_backbone.index)
        
        chunk_df = pile.manifest.loc[chunk_indices]
        needs_backbone = chunk_df.loc[chunk_df.backbone_filename.isnull()]
        assert len(needs_backbone) == 0, needs_backbone
        pile.save()

    # ProteinMPNN
    needs_seq = chunk_df.loc[chunk_df.seq.isnull()]
    if len(needs_seq) > 0:
        print("Running proteinmpnn")
        run_proteinmpnn_chunk(args, client, spec, pile, needs_seq.index)
        chunk_df = pile.manifest.loc[chunk_indices]
        needs_seq = chunk_df.loc[chunk_df.seq.isnull()]
        assert len(needs_seq) == 0, needs_seq
        pile.save()

    # Structure predictors
    for predictor in ["af2", "omegafold"]:
        needs_prediction = chunk_df.loc[chunk_df[predictor + "_filename"].isnull()]
        if len(needs_prediction) > 0:
            print("Running", predictor)
            run_structure_predictor(
                predictor,
                args,
                client,
                pile,
                needs_prediction.index)
            pile.save()

        chunk_df = pile.manifest.loc[chunk_indices]
        needs_prediction = chunk_df.loc[chunk_df[predictor + "_filename"].isnull()]
        assert len(needs_prediction) == 0, needs_prediction

    # Metrics
    chunk_df = pile.manifest.loc[chunk_indices]
    needs_metrics = chunk_df.loc[
        chunk_df.metrics_dict.isnull()
    ]
    if len(needs_metrics) > 0:
        print("Computing metrics")
        metrics_list = []
        for idx, row in tqdm.tqdm(needs_metrics.iterrows(), total=len(needs_metrics)):
            design = pile.get_design(spec, name=idx)
            metrics = spec.get_metrics(design)
            metrics_list.append(metrics)
        pile.manifest.loc[needs_metrics.index, "metrics_dict"] = metrics_list
        pile.save()


def run_structure_predictor(predictor_name, args, client, pile, chunk_indices):
    runner = None
    if predictor_name == "af2":
        runner = common.get_runner(
            client,
            proteopt.alphafold.AlphaFold,
            max_length=int(pile.manifest.loc[chunk_indices].seq.str.len().max()),
            model_name="model_4_ptm",
            num_recycle=0,
            amber_relax=False)
    elif predictor_name == "omegafold":
        runner = common.get_runner(
            client,
            proteopt.omegafold.OmegaFold,
        )
    else:
        raise NotImplementedError(predictor_name)

    results = runner.run_multiple(
        pile.manifest.loc[chunk_indices].seq.values,
        show_progress=True,
        items_per_request=args.items_per_request)

    for ((name, row), prediction) in zip(pile.manifest.loc[chunk_indices].iterrows(), results):
        filename = f"{name}.{predictor_name}.pdb"
        filepath = os.path.join(args.intermediates_dir, filename)
        prody.writePDB(filepath, prediction)
        print("Wrote", filepath)
        pile.manifest.loc[name, f"{predictor_name}_filename"] = filename
        info = {}
        if predictor_name == "af2":
            info = {
                "ptm": prediction.getData("af2_ptm").mean(),
                "mean_plddt": prediction.getData("af2_plddt").mean(),
                "plddt": [round(x) for x in prediction.ca.getData("af2_plddt")],
            }
        pile.manifest.loc[[name], "%s_dict" % predictor_name] = [info]
        pile.save(skip_seconds=5.0)


def sample_chunk_grouped_by_params_or_design_backbone_filename(df, chunksize):
    chunk_indices = set()
    if len(df) < chunksize:
        chunk_indices = set(df.index)
    else:
        while len(chunk_indices) < chunksize:
            if df.backbone_filename.isnull().any():
                chunk_indices.update(
                    df.loc[
                        df.params_dict == df.sample().iloc[0].params_dict
                    ].index)
            else:
                chunk_indices.update(
                    df.loc[
                        df.backbone_filename == df.sample().iloc[0].backbone_filename
                    ].index)
    chunk_indices = sorted(chunk_indices)
    assert len(chunk_indices) > 0
    return chunk_indices


def run_proteinmpnn_chunk(args, client, spec, pile, chunk_indices, sampling_temp=0.1):
    proteinmpnn_runner = common.get_runner(
        client, proteopt.proteinmpnn.ProteinMPNN, sampling_temp=sampling_temp)
    proteinmpnn_work = []
    proteinmpnn_work_design_indices = []
    problems = {}
    for d, big_sub_df in pile.manifest.loc[chunk_indices].groupby("backbone_filename"):
        if len(big_sub_df) > client.max_parallelism * 10:
            # Break into chunks
            sub_dfs = numpy.array_split(big_sub_df, client.max_parallelism)
        else:
            sub_dfs = [big_sub_df]

        handle = pile.load_pdb(d).select("chain A").copy()
        params_dict, = pile.manifest.loc[big_sub_df.index, "params_dict"].unique()
        problem = common.get_problem(spec, params_dict).get_first_chain()
        problem.annotate_solution(handle)
        problems[params_dict] = problem

        for sub_df in sub_dfs:
            proteinmpnn_work.append({
                "structure": handle,
                "fixed": handle.select("constrained_by_sequence"),
                "num": len(sub_df) + 1,  # one extra to allow for duplicates
            })
            proteinmpnn_work_design_indices.append(sub_df.index.values)

    proteinmpnn_results = proteinmpnn_runner.run_multiple(
        proteinmpnn_work, items_per_request=args.items_per_request, show_progress=True)

    indices_to_rerun = set()
    for (indices, generated) in zip(
            proteinmpnn_work_design_indices, tqdm.tqdm(proteinmpnn_results)):
        seqs = generated.sort_values("score", ascending=False).seq_A
        if sampling_temp < 1.0:
            # To avoid infinite loops for weird cases, we accept duplicates
            # once the sampling temp reaches 1.0
            seqs = seqs[
                ~seqs.isin(pile.manifest.seq.dropna())
            ].drop_duplicates()
        seqs = seqs.head(len(indices)).values
        params_dict, = pile.manifest.loc[indices, "params_dict"].unique()
        problem = problems[params_dict]
        for seq in seqs:
            assert problem.check_solution_sequence_is_valid(seq), (problem, seq)
        pile.manifest.loc[indices[:len(seqs)], "seq"] = seqs
        pile.save(skip_seconds=5.0)

        if len(seqs) < len(indices):
            indices_to_rerun.update(indices[len(seqs):])

    if indices_to_rerun:
        indices_to_rerun = sorted(indices_to_rerun)
        new_sampling_temp = min(sampling_temp * 1.1, 1.0)
        print(
            f"Rerunning proteinmpnn on {len(indices_to_rerun)}/{len(chunk_indices)} "
            f"at sampling temp {new_sampling_temp} (was {sampling_temp}) to find "
            f"non duplicative sequences")
        run_proteinmpnn_chunk(
            args, client, spec, pile, indices_to_rerun, sampling_temp=new_sampling_temp)


def run_rfdiffusion_chunk(args, client, spec, pile, chunk_indices):
    runner = common.get_runner(
            client,
            proteopt.rfdiffusion_motif.RFDiffusionMotif,
            local_conf={
                "models_dir": args.rfdiffusion_weights_dir,
            })

    work = []
    work_indices = []

    # Because groupby of a frozendict seems broken
    pile.manifest["params_dict_tuples"] = pile.manifest.params_dict.map(
        lambda d: tuple(sorted(d.items()))
    )

    for params, sub_df in pile.manifest.loc[chunk_indices].groupby("params_dict_tuples"):
        problem = spec.get_problem(dict(params))
        work_indices.append(sub_df.index.values)
        work.append({
            "problem": problem,
        })
    pile.manifest["params_dict_tuples"]

    results = runner.run_multiple(
        work, items_per_request=args.items_per_request, show_progress=True)

    for (indices, generated) in zip(work_indices, tqdm.tqdm(results)):
        assert len(generated) == 1, len(generated)
        id = hashlib.sha1(str(time.time()).encode()).hexdigest()
        filename = f"backbone.{id}.pdb"
        filepath = os.path.join(pile.intermediates_dir, filename)
        structure = generated.iloc[0].structure
        prody.writePDB(filepath, structure)
        print("Wrote", filepath)
        pile.manifest.loc[indices, "backbone_filename"] = filename
        pile.save(skip_seconds=5.0)
