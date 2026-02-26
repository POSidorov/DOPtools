#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright 2022-2025 Pavel Sidorov <pavel.o.sidorov@gmail.com> This
#  file is part of DOPTools repository.
#
#  DOPtools is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.

import argparse
import glob
import os
import pickle
import warnings
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from doptools.optimizer.config import get_raw_model

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)


import argparse
import glob
import os
import pickle
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd

from doptools.optimizer.config import get_raw_model


class Rebuilder:
    def __init__(
        self,
        file: str = None,
        folders: List[str] = None,
        desc_folder: str = None,
        ensemble: int = 1,
        score_threshold=0.5,
    ):
        self.file = file
        self.folders = folders
        self.desc_folder = desc_folder
        if self.file is None and self.folders is None:
            raise ValueError(
                "At least one file or folder should be given to rebuild models"
            )
        self.ensemble = ensemble
        self.score_threshold = score_threshold
        self.prop = ""
        self.model = None
        self.trained = False

    def gather_trials(self, trials="all"):
        trial_files = []
        if self.folders is not None:
            for f in self.folders:
                trial_files.append(os.path.join(f, "trials." + trials))
        elif self.file is not None:
            trial_files.append(self.file)

        full_df = pd.concat([pd.read_table(f, sep=" ") for f in trial_files])
        full_df = full_df[full_df["score"] >= self.score_threshold]
        self.prop = (
            pd.read_table(
                trial_files[0][:-5] + "." + str(full_df.iloc[0].trial) + "/stats",
                sep=" ",
            )
            .iloc[0]["stat"]
            .split(".")[0]
        )
        return full_df

    def rebuild(self, one_per_descriptor=False):
        trials = self.gather_trials()
        trials = trials.sort_values(by="score", ascending=False)
        models = []
        selected_descs = []

        for i, row in trials.iterrows():
            if len(models) >= self.ensemble:
                break
            if one_per_descriptor and row.desc in selected_descs:
                print(row.desc)
                continue
            else:
                pipeline_steps = []
                desc_name = row["desc"]
                if os.path.isdir(
                    os.path.join(self.desc_folder, desc_name.split("_")[0])
                ):
                    desc_file = os.path.join(
                        self.desc_folder,
                        desc_name.split("_")[0],
                        self.prop + "." + desc_name + ".pkl",
                    )
                else:
                    desc_file = os.path.join(
                        self.desc_folder, self.prop + "." + desc_name + ".pkl"
                    )

                with open(desc_file, "rb") as f:
                    desc_calculator = pickle.load(f)
                pipeline_steps.append(("descriptor_calculator", desc_calculator))

                if row["scaling"] == "scaled":
                    pipeline_steps.append(("scaler", MinMaxScaler()))

                pipeline_steps.append(("variance", VarianceThreshold()))

                params = row[
                    list(row.index)[list(row.index).index("method") + 1 :]
                ].to_dict()
                for k, p in params.items():
                    if pd.isnull(p):
                        params[k] = None
                method = row["method"]
                model = get_raw_model(method, params)
                pipeline_steps.append(("model", model))

                models.append(Pipeline(pipeline_steps))
                selected_descs.append(desc_name)
        if len(models) == 1:
            self.model = models[0]
        else:
            self.model = ConsensusModel(models)

    def train(self, train_set, train_prop, smiles_column=None):
        if self.model is None:
            raise AttributeError(
                "The model has not been created yet. Use rebuild function first."
            )

        if isinstance(train_set, str):
            if train_set.endswith("xlsx") or train_set.endswith("xls"):
                train_data = pd.read_excel(train_set)
            elif train_set.endswith("csv"):
                train_data = pd.read_table(train_set)
            if smiles_column is not None or isinstance(models[0][0], ComplexFragmentor):
                x_train = train_data[smiles_column]
            else:
                x_train = train_data
        elif isinstance(train_set, Iterable):
            x_train = train_set

        self.model.fit(x_train, train_prop)
        self.trained = True

    def save_model(self, save_dest):
        if not os.path.exists(save_dest):
            os.makedirs(
                save_dest, exist_ok=True
            )  # exist_ok is useful when several processes try to create the folder at the same time
            print("The output directory {} created".format(save_dest))
        if self.model is None:
            raise AttributeError(
                "The model has not been created yet. Use rebuild function first."
            )
        if isinstance(self.model, ConsensusModel):
            filename = ".".join(
                [
                    "consensus",
                    "trained",
                    datetime.now().strftime("%Y-%m-%d-%H-%M"),
                    "pkl",
                ]
            )
        else:
            filename = ".".join(
                [
                    self.model[0].short_name,
                    self.model[-1].__class__.__name__,
                    (not self.trained) * "un" + "trained",
                    datetime.now().strftime("%Y-%m-%d-%H-%M"),
                    "pkl",
                ]
            )
        with open(os.path.join(save_dest, filename), "wb") as f:
            pickle.dump(self.model, f, pickle.HIGHEST_PROTOCOL)

    def apply(self, test_set, smiles_column=None):
        if isinstance(test_set, str):
            if test_set.endswith("xlsx") or test_set.endswith("xls"):
                test_data = pd.read_excel(test_set)
            elif test_set.endswith("csv"):
                test_data = pd.read_table(test_set)
            if smiles_column is not None or isinstance(models[0][0], ComplexFragmentor):
                x_test = test_data[smiles_column]
            else:
                x_test = test_data
        elif isinstance(test_set, Iterable):
            x_test = test_set
        results = self.model.predict(x_test)
        return results

    def rebuild_save(self, save_dest, one_per_descriptor=False):
        self.rebuild(one_per_descriptor)
        self.save_model(save_dest)

    def rebuild_train_save(
        self,
        save_dest,
        train_set,
        train_prop,
        smiles_column=None,
        one_per_descriptor=False,
    ):
        self.rebuild(one_per_descriptor)
        self.train(train_set, train_prop, smiles_column)
        self.save_model(save_dest, trained=True)

    def rebuild_train_apply(
        self,
        train_set,
        train_prop,
        test_set,
        smiles_column=None,
        one_per_descriptor=False,
    ):
        self.rebuild(one_per_descriptor)
        self.train(train_set, train_prop, smiles_column)
        results = self.apply(test_set, smiles_column)
        return results

    def save_self(self, save_dest):
        with open(save_dest, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def rebuilder():
    parser = argparse.ArgumentParser(
        prog="Optimized model rebuilder",
        description="Rebuilds the model from the optimized trial parameters,\nsaving it as an UNTRAINED pipeline in pickle",
    )
    parser.add_argument(
        "-d",
        "--descdir",
        required=True,
        help="the folder containing descriptor files and calculators. Can contain folders separated by descriptor type",
    )
    parser.add_argument(
        "-f", "--fileinput", help='the "trials.all" or "trails.best" file.'
    )
    parser.add_argument(
        "-m",
        "--modeldir",
        help='the folder containing model output files. Should contain "trials.all" file.',
    )
    parser.add_argument(
        "-o", "--outdir", required=True, help="the output folder for the models."
    )
    parser.add_argument(
        "-e",
        "--ensemble",
        type=int,
        deafult=1,
        help="the number of models that would be taken for an ensemble. Default 1 (non-ensemble).",
    )
    parser.add_argument(
        "-e",
        "--ensemble",
        action="store_true",
        help="toggle to indicate that only one model per descriptor type is taken into ensemble",
    )

    args = parser.parse_args()
    descdir = args.descdir
    modeldir = args.modeldir
    number = args.number
    outdir = args.outdir

    if os.path.exists(outdir):
        print(
            "The output directory {} already exists. The data may be overwritten".format(
                outdir
            )
        )
    else:
        os.makedirs(outdir)
        print("The output directory {} created".format(outdir))

    pipeline, trial = rebuild_from_file(descdir, modeldir, number)

    modelfile_name = "_".join([trial["method"], "trial" + str(number), trial["desc"]])
    with open(os.path.join(outdir, modelfile_name + ".pkl"), "wb") as f:
        pickle.dump(pipeline, f)


__all__ = ["rebuild_from_file"]
