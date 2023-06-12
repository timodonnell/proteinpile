import os
import json
import shutil
import hashlib
import time
import logging

import numpy
import pandas
from frozendict import frozendict

from .common import load_pdb
from .design import Design

class Pile(object):
    COLUMNS = [
        "method",
        "params_dict",
        "backbone_filename",
        "seq",
        "af2_filename",
        "af2_dict",
        "af2_templated_filename",
        "af2_templated_dict",
        "omegafold_filename",
        "omegafold_dict",
        "metrics_dict",
        "provenance_dict",
        "serial_number",
    ]

    @classmethod
    def blank(cls, manifest_path, intermediates_dir=None):
        obj = cls(None, intermediates_dir=intermediates_dir)
        obj.manifest_path = manifest_path
        obj.original_manifest_hash = None
        obj.manifest = pandas.DataFrame(columns=cls.COLUMNS)
        return obj

    def __init__(self, manifest_path, intermediates_dir=None):
        if manifest_path is not None:
            self.manifest_path = manifest_path
            self.original_manifest_hash = hashlib.sha1(
                open(self.manifest_path, "rb").read()).hexdigest()
            if manifest_path.endswith(".csv") or manifest_path.endswith(".csv.gz"):
                self.manifest = load_manifest_csv(manifest_path)
            else:
                raise NotImplementedError("Unknown format: %s" % manifest_path)
            for col in self.COLUMNS:
                if col not in self.manifest:
                    self.manifest[col] = None
                if col == "serial_number":
                    self.manifest.loc[self.manifest[col].isnull(), col] = 0.0
                elif col == "method":
                    self.manifest.loc[self.manifest[col].isnull(), col] = "rfdiffusion_motif-proteinmpnn"
                else:
                    self.manifest.loc[self.manifest[col].isnull(), col] = None
            self.manifest = self.manifest[self.COLUMNS]
            numpy.testing.assert_equal(set(self.manifest.columns), set(self.COLUMNS))
            numpy.testing.assert_equal(len(set(self.manifest.index)), len(self.manifest))
        self.intermediates_dir = intermediates_dir
        self.last_save_time = 0

    def write_manifest(self, manifest_path):
        self.manifest.loc[self.manifest.serial_number.isnull(), "serial_number"] = (
            self.manifest.serial_number.max() + 1
        )
        if manifest_path.endswith(".csv") or manifest_path.endswith(".csv.gz"):
            write_manifest_csv(manifest_path, self.manifest)
        else:
            raise NotImplementedError("Unknown format: %s" % manifest_path)

    def save(self, backup_dir="/tmp/manifest-backups", skip_seconds=None):
        if skip_seconds and time.time() - skip_seconds <= self.last_save_time:
            # Don't save since we already saved very recently.
            return

        if backup_dir is not None and self.original_manifest_hash is not None:
            if not os.path.exists(backup_dir):
                os.mkdir(backup_dir)
            backup_path = os.path.join(
                backup_dir, self.original_manifest_hash + "_" + os.path.basename(self.manifest_path))
            shutil.copy(self.manifest_path, backup_path)
            print("Wrote backup to ", backup_path)
        self.write_manifest(self.manifest_path)
        self.last_save_time = time.time()

    def summarize_metrics(self):
        metrics_df = pandas.DataFrame.from_records(
            self.manifest.metrics_dict.dropna().values)
        if len(metrics_df) == 0:
            print("< No metrics >")
            return
        print("Metrics summary [n=%d]" % len(metrics_df))
        for col in metrics_df.columns:
            v = metrics_df[col]
            print(f"\t{col}\tmin={v.min()} median={v.median()} max={v.max()}")

    def get_path(self, filename):
        return os.path.join(self.intermediates_dir, filename)

    def load_pdb(self, filename):
        return load_pdb(self.get_path(filename))

    def get_designs(self, spec, names=None):
        abbreviated_names = (
                self.manifest.index.to_series().str.slice(0, 10) +
                self.manifest.index.to_series().str.slice(-6)).str.replace(".", "", regex=False)

        if abbreviated_names.value_counts().max() > 1:
            logging.warning(
                "Abbreviated names not unique, falling back to full names: %s" %
                abbreviated_names.value_counts().head(10))
            abbreviated_names = self.manifest.index.to_series()

        if names is None:
            names = self.manifest.index
        return [
            self.get_design(spec, name, abbreviated_name=abbreviated_names[name])
            for name in names
        ]

    def get_design(self, spec, name, abbreviated_name=None):
        row = self.manifest.loc[name]
        return Design(
            spec=spec,
            intermediates_dir=self.intermediates_dir,
            row=row,
            abbreviated_name=abbreviated_name)


def load_manifest_csv(filename):
    df = pandas.read_csv(filename, index_col=0)
    for col in df.columns:
        if col.endswith("_json"):
            df[col.replace("_json", "_dict")] = df[col].map(
                lambda s:
                    None if type(s) == float and numpy.isnan(s)
                    else frozendict(json.loads(s)))
            del df[col]
    print("Read input data with shape:", *df.shape)
    print(df.head(3))
    return df


def write_manifest_csv(filename, design_df):
    write_df = design_df.copy()
    for col in write_df.columns:
        if col.endswith("_dict"):
            write_df[col.replace("_dict", "_json")] = write_df[col].map(json.dumps)
            del write_df[col]
    write_df.to_csv(filename)
    print("Wrote: %d rows" % len(write_df), filename)
