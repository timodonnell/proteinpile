import proteopt.alignment


def required_structure_predictors():
    return ["af2", "omegafold"]


def get_metrics(design):
    raw = design.get_structure("backbone").select("chain A")
    result = {}
    for predictor in ["af2", "omegafold"]:
        result.update(design.problem.get_first_chain().evaluate_solution(
            design.get_structure(predictor), prefix=f"{predictor}_"))
        result[
            f"{predictor}_ca_rmsd_to_design"] = proteopt.alignment.smart_align(
            design.get_structure(predictor).ca, raw.ca, ).rmsd
    result["ca_rmsd_af2_vs_omegafold"] = proteopt.alignment.smart_align(
        design.get_structure("omegafold").ca,
        design.get_structure("af2").ca, ).rmsd
    return result


def loss(design):
    return (
        design.metrics_dict["af2_motif_0_all_atom_rmsd"],
        design.metrics_dict["omegafold_motif_0_all_atom_rmsd"],
    )


def directions():
    return ("minimize", "minimize")