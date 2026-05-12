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
import os
import pickle
import warnings
from datetime import datetime
from typing import Any, Iterable, List, Optional, Tuple

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from doptools.chem.chem_features import ComplexFragmentor
from doptools.estimators.consensus import ConsensusModel
from doptools.optimizer.config import get_raw_model

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)


class Rebuilder:
    def __init__(
        self,
        file: Optional[str] = None,
        folders: Optional[List[str]] = None,
        desc_folder: Optional[str] = None,
        ensemble: int = 1,
        score_threshold: float = 0.5,
    ) -> None:
        self.file: Optional[str] = file
        self.folders: Optional[List[str]] = folders
        self.desc_folder: Optional[str] = desc_folder
        if self.file is None and self.folders is None:
            raise ValueError(
                "At least one file or folder should be given to rebuild models"
            )
        self.ensemble: int = ensemble
        self.score_threshold: float = score_threshold
        self.prop: str = ""
        self.model: Optional[Any] = None
        self.trained: bool = False

    def gather_trials(self, trials: str = "all") -> pd.DataFrame:
        trial_files: List[str] = []
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

    def rebuild(self, one_per_descriptor: bool = False) -> None:
        if self.desc_folder is None:
            raise ValueError("desc_folder must be provided to rebuild models.")
        trials = self.gather_trials()
        trials = trials.sort_values(by="score", ascending=False)
        models: List[Any] = []
        selected_descs: List[str] = []

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

    def train(
        self,
        train_set: Any,
        train_prop: Any,
        smiles_column: Optional[str] = None,
    ) -> None:
        if self.model is None:
            raise AttributeError(
                "The model has not been created yet. Use rebuild function first."
            )

        if isinstance(train_set, str):
            if train_set.endswith("xlsx") or train_set.endswith("xls"):
                train_data = pd.read_excel(train_set)
            elif train_set.endswith("csv"):
                train_data = pd.read_table(train_set)
            descriptor = (
                self.model.pipelines[0][0]
                if isinstance(self.model, ConsensusModel)
                else self.model[0]
            )
            if smiles_column is not None or isinstance(descriptor, ComplexFragmentor):
                x_train = train_data[smiles_column]
            else:
                x_train = train_data
        elif isinstance(train_set, Iterable):
            x_train = train_set

        self.model.fit(x_train, train_prop)
        self.trained = True

    def save_model(self, save_dest: str, trained: Optional[bool] = None) -> None:
        if trained is not None:
            self.trained = trained
        if not os.path.exists(save_dest):
            os.makedirs(save_dest, exist_ok=True)
            # exist_ok helps when several processes try to create the folder at once
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

    def apply(self, test_set: Any, smiles_column: Optional[str] = None) -> Any:
        if self.model is None:
            raise AttributeError(
                "The model has not been created yet. Use rebuild function first."
            )
        if isinstance(test_set, str):
            if test_set.endswith("xlsx") or test_set.endswith("xls"):
                test_data = pd.read_excel(test_set)
            elif test_set.endswith("csv"):
                test_data = pd.read_table(test_set)
            descriptor = (
                self.model.pipelines[0][0]
                if isinstance(self.model, ConsensusModel)
                else self.model[0]
            )
            if smiles_column is not None or isinstance(descriptor, ComplexFragmentor):
                x_test = test_data[smiles_column]
            else:
                x_test = test_data
        elif isinstance(test_set, Iterable):
            x_test = test_set
        results = self.model.predict(x_test)
        return results

    def rebuild_save(self, save_dest: str, one_per_descriptor: bool = False) -> None:
        self.rebuild(one_per_descriptor)
        self.save_model(save_dest)

    def rebuild_train_save(
        self,
        save_dest: str,
        train_set: Any,
        train_prop: Any,
        smiles_column: Optional[str] = None,
        one_per_descriptor: bool = False,
    ) -> None:
        self.rebuild(one_per_descriptor)
        self.train(train_set, train_prop, smiles_column)
        self.save_model(save_dest, trained=True)

    def rebuild_train_apply(
        self,
        train_set: Any,
        train_prop: Any,
        test_set: Any,
        smiles_column: Optional[str] = None,
        one_per_descriptor: bool = False,
    ) -> Any:
        self.rebuild(one_per_descriptor)
        self.train(train_set, train_prop, smiles_column)
        results = self.apply(test_set, smiles_column)
        return results

    def save_self(self, save_dest: str) -> None:
        with open(save_dest, "wb") as f:
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)


def rebuild_from_file(
    descdir: str, modeldir: str, number: int
) -> Tuple[Any, dict[str, Any]]:
    raise NotImplementedError(
        "rebuild_from_file is not implemented. Use Rebuilder.rebuild for now."
    )


def rebuilder() -> None:
    parser = argparse.ArgumentParser(
        prog="Optimized model rebuilder",
        description=(
            "Rebuilds the model from the optimized trial parameters, saving it as "
            "an UNTRAINED pipeline in pickle"
        ),
    )
    parser.add_argument(
        "-d",
        "--descdir",
        required=True,
        help=(
            "the folder containing descriptor files and calculators. Can contain "
            "folders separated by descriptor type"
        ),
    )
    parser.add_argument(
        "-f", "--fileinput", help='the "trials.all" or "trails.best" file.'
    )
    parser.add_argument(
        "-m",
        "--modeldir",
        help=(
            'the folder containing model output files. Should contain "trials.all" '
            "file."
        ),
    )
    parser.add_argument(
        "-o", "--outdir", required=True, help="the output folder for the models."
    )
    parser.add_argument(
        "-e",
        "--ensemble",
        type=int,
        deafult=1,
        help=(
            "the number of models that would be taken for an ensemble. Default 1 "
            "(non-ensemble)."
        ),
    )
    parser.add_argument(
        "-e",
        "--ensemble",
        action="store_true",
        help=(
            "toggle to indicate that only one model per descriptor type is taken "
            "into ensemble"
        ),
    )

    args = parser.parse_args()
    descdir = args.descdir
    modeldir = args.modeldir
    number = args.number
    outdir = args.outdir

    if os.path.exists(outdir):
        print(
            "The output directory {} already exists. The data may be "
            "overwritten".format(outdir)
        )
    else:
        os.makedirs(outdir)
        print("The output directory {} created".format(outdir))

    pipeline, trial = rebuild_from_file(descdir, modeldir, number)

    modelfile_name = "_".join([trial["method"], "trial" + str(number), trial["desc"]])
    with open(os.path.join(outdir, modelfile_name + ".pkl"), "wb") as f:
        pickle.dump(pipeline, f)


__all__ = ["rebuild_from_file"]
