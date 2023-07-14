import os
from . import common


class Design(object):
    def __init__(self, spec, intermediates_dir, row, abbreviated_name=None):
        if abbreviated_name is None:
            abbreviated_name = row.name.replace(".", "")

        self.abbreviated_name = abbreviated_name

        self.spec = spec
        self.intermediates_dir = intermediates_dir

        # Lazy computed properties
        self._problem = None
        self._structures = {}

        # Pull out fields from row
        self.name = row.name
        self.method = row["method"]
        self.backbone_filename = row["backbone_filename"]
        self.seq = row["seq"]
        self.af2_filename = row["af2_filename"]
        self.af2_templated_filename = row["af2_templated_filename"]
        self.omegafold_filename = row["omegafold_filename"]
        self.serial_number = row["serial_number"]
        self.params_dict = row["params_dict"]
        self.metrics_dict = row["metrics_dict"]
        self.af2_dict = row["af2_dict"]
        self.af2_templated_dict = row["af2_dict"]
        self.omegafold_dict = row["omegafold_dict"]
        self.provenance_dict = row["provenance_dict"]

    @property
    def problem(self):
        if self._problem is None:
            self._problem = common.get_problem(self.spec, self.params_dict)
        return self._problem

    def get_structure_path(self, predictor):
        if predictor == "backbone":
            if self.backbone_filename == "NA":
                raise ValueError("Structure prediction %s was not computed" % predictor)
            return os.path.join(self.intermediates_dir, self.backbone_filename)
        elif predictor == "af2":
            if self.af2_filename == "NA":
                raise ValueError("Structure prediction %s was not computed" % predictor)
            return os.path.join(self.intermediates_dir, self.af2_filename)
        elif predictor == "af2_templated":
            if self.af2_templated_filename == "NA":
                raise ValueError("Structure prediction %s was not computed" % predictor)
            return os.path.join(self.intermediates_dir, self.af2_templated_filename)
        elif predictor == "omegafold":
            if self.omegafold_filename == "NA":
                raise ValueError("Structure prediction %s was not computed" % predictor)
            return os.path.join(self.intermediates_dir, self.omegafold_filename)
        else:
            raise ValueError("Unknown structure predictor: %s" % predictor)

    def get_structure(self, predictor):
        try:
            return self._structures[predictor]
        except KeyError:
            pass
        result = common.load_pdb(self.get_structure_path(predictor))
        problem = self.problem
        if predictor != "backbone":
            problem = problem.get_first_chain()
        problem.annotate_solution(result)
        self._structures[predictor] = result
        return result

