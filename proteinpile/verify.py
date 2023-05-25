import numpy
import tqdm

from . import common

def add_args(parser):
    pass


def handle_verify(args, pile, spec):
    if 'metrics_dict' in pile.manifest:
        pile.summarize_metrics()

    df = pile.manifest
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        if row.seq is not None:
            problem = common.get_problem(spec, row.params_dict)
            problem.check_solution_sequence_is_valid(row.seq)
            if row.backbone_filename is not None:
                backbone = pile.load_pdb(row.backbone_filename)
                numpy.testing.assert_equal(len(backbone.select("chain A").ca), len(row.seq))
            if row.af2_filename is not None:
                pred = pile.load_pdb(row.af2_filename)
                numpy.testing.assert_equal(pred.ca.getSequence(), row.seq)
            if row.omegafold_filename is not None:
                pred = pile.load_pdb(row.omegafold_filename)
                numpy.testing.assert_equal(pred.ca.getSequence(), row.seq)



