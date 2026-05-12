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

import copy
import glob
import json
import multiprocessing as mp
import os
import warnings
from functools import partial
from typing import Any, Dict, MutableMapping, Optional, Tuple, Union

import numpy as np
import optuna
import pandas as pd
from optuna.study import StudyDirection
from pandas import DataFrame
from scipy.sparse import issparse
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
)
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
from sklearn.utils import shuffle
from timeout_decorator import TimeoutError, timeout_decorator

from doptools.optimizer.config import get_raw_model, suggest_params
from doptools.optimizer.utils import r2, rmse

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=DeprecationWarning)

optuna.logging.set_verbosity(optuna.logging.WARNING)


class TopNPatienceCallback:
    def __init__(self, patience: int, leaders: int = 1) -> None:
        self.patience: int = patience
        self.leaders: int = leaders
        self._leaders_unchanged_steps: int = 0
        self._previous_leaders: Tuple[int, ...] = ()

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        if len(study.trials_dataframe()) < self.leaders:
            return

        if study.direction == StudyDirection.MAXIMIZE:
            new_leaders = tuple(
                study.trials_dataframe()
                .sort_values(by="value", ascending=False)
                .head(self.leaders)
                .number
            )
        else:
            new_leaders = tuple(
                study.trials_dataframe()
                .sort_values(by="value", ascending=True)
                .head(self.leaders)
                .number
            )

        if new_leaders == self._previous_leaders:
            self._leaders_unchanged_steps += 1
        else:
            self._leaders_unchanged_steps = 0
            self._previous_leaders = new_leaders

        if self._leaders_unchanged_steps >= self.patience:
            study.stop()


def collect_data(
    datadir: str, task: str, fmt: str = "svm"
) -> Tuple[Dict[str, Any], DataFrame]:
    desc_dict: Dict[str, Any] = {}
    y: Dict[str, Any] = {}
    for f in glob.glob(os.path.join(datadir, "*." + fmt)):
        propname = f.split(os.sep)[-1].split(".")[0]
        name = f.split(os.sep)[-1][len(propname) + 1 : -4]
        if fmt == "svm":
            desc_dict[name], y[propname] = load_svmlight_file(f)
        elif fmt == "csv":
            data = pd.read_table(f)
            y[propname] = data[propname]
            col_idx = list(data.columns).index(propname)
            desc_dict[name] = data.iloc[:, col_idx + 1 :]
    if task.endswith("C"):
        return desc_dict, pd.DataFrame(y, dtype=int)
    else:
        return desc_dict, pd.DataFrame(y)


def calculate_scores(task: str, obs: DataFrame, pred: DataFrame) -> DataFrame:
    def create_row(
        task: str, stat_name: str, x: pd.Series, y: pd.Series
    ) -> Dict[str, Union[str, float]]:
        if task == "R":
            return {
                "stat": stat_name,
                "R2": r2(x, y),
                "RMSE": rmse(x, y),
                "MAE": mae(x, y),
            }
        elif task == "C" and len(set(x)) == 2:
            return {
                "stat": stat_name,
                "ROC_AUC": roc_auc_score(x, y),
                "ACC": accuracy_score(x, y),
                "BAC": balanced_accuracy_score(x, y),
                "F1": f1_score(x, y),
                "MCC": matthews_corrcoef(x, y),
            }
        elif task == "C" and len(set(x)) > 2:
            return {
                "stat": stat_name,
                "ROC_AUC": roc_auc_score(
                    LabelBinarizer().fit_transform(x),
                    LabelBinarizer().fit_transform(y),
                    multi_class="ovr",
                ),
                "ACC": accuracy_score(x, y),
                "BAC": balanced_accuracy_score(x, y),
                "F1": f1_score(x, y, average="macro"),
                "MCC": matthews_corrcoef(x, y),
            }
        raise ValueError("Unknown task type")

    if task == "R":
        score_df = pd.DataFrame(columns=["stat", "R2", "RMSE", "MAE"])
    elif task == "C":
        score_df = pd.DataFrame(columns=["stat", "ROC_AUC", "ACC", "BAC", "F1", "MCC"])
    else:
        raise ValueError("Unknown task type")

    for c in obs.columns:
        preds_partial = pred[[d for d in pred.columns if c + ".predicted." in d]]
        for p in preds_partial.columns:
            added_row = create_row(task, p, obs[c], preds_partial[p])
            score_df = pd.concat(
                [pd.DataFrame(added_row, index=[0]), score_df.loc[:]]
            ).reset_index(drop=True)
        if task == "R":
            added_row = create_row(
                task, c + ".consensus", obs[c], preds_partial.mean(axis=1)
            )
        elif task == "C":
            added_row = create_row(
                task, c + ".consensus", obs[c], np.round(preds_partial.mean(axis=1))
            )
        score_df = pd.concat(
            [pd.DataFrame(added_row, index=[0]), score_df.loc[:]]
        ).reset_index(drop=True)
    return score_df


def objective_study(
    storage: MutableMapping[int, Dict[str, Any]],
    results_detailed: MutableMapping[int, Dict[str, Any]],
    trial: optuna.trial.Trial,
    x_dict: Dict[str, Any],
    y: DataFrame,
    outdir: str,
    method: str,
    ntrials: int,
    cv_splits: int,
    cv_repeats: int,
    jobs: int,
    tmout: int,
    earlystop: Tuple[int, int],
    write_output: bool = True,
) -> float:
    n = trial.number
    if write_output and not os.path.exists(os.path.join(outdir, "trial." + str(n))):
        os.mkdir(os.path.join(outdir, "trial." + str(n)))
    res_pd = pd.DataFrame(columns=["data_index"])
    res_pd["data_index"] = np.arange(1, len(y) + 1, step=1).astype(int)

    desc = trial.suggest_categorical("desc_type", list(x_dict.keys()))
    scaling = trial.suggest_categorical("scaling", ["scaled", "original"])

    X = x_dict[desc]
    if issparse(X):
        X = X.toarray()

    if scaling == "scaled":
        mms = MinMaxScaler()
        X = mms.fit_transform(X)

    X = VarianceThreshold().fit_transform(X)

    params = suggest_params(trial, method)
    model = get_raw_model(method, params)

    Y = np.array(y[y.columns[0]])
    if method.endswith("C"):
        LE = LabelEncoder()
        Y = LE.fit_transform(Y)

    for r in range(cv_repeats):
        Y = pd.Series(Y)
        X, Y = shuffle(X, Y)
        shuffle_indices = Y.index

        preds = cross_val_predict(model, X, Y, cv=KFold(cv_splits))
        # if len(y.columns)<2:
        if method.endswith("C"):
            preds = LE.inverse_transform(preds)
            preds_proba = cross_val_predict(
                model, X, Y, cv=KFold(cv_splits), method="predict_proba"
            )
            for i, c in enumerate(y.columns):
                res_pd[c + ".observed"] = y[c]
                res_pd[c + ".predicted.class.repeat" + str(r + 1)] = pd.Series(
                    preds, index=shuffle_indices
                ).sort_index()
                for j in range(preds_proba.shape[1]):
                    res_pd[
                        c + ".predicted_prob.class_" + str(j) + ".repeat" + str(r + 1)
                    ] = pd.Series(preds_proba[:, j], index=shuffle_indices).sort_index()
        else:
            # preds = preds.reshape((-1, 1))
            for i, c in enumerate(y.columns):
                res_pd[c + ".observed"] = y[c]
                res_pd[c + ".predicted.repeat" + str(r + 1)] = pd.Series(
                    preds, index=shuffle_indices
                ).sort_index()

    score_df = calculate_scores(method[-1], y, res_pd)

    fit_scores: Dict[str, Union[str, float]] = {}
    model.fit(X, Y)
    fit_preds = model.predict(X)
    if method.endswith("R"):
        fit_scores = {
            "stat": "fit",
            "R2": r2(Y, fit_preds),
            "RMSE": rmse(Y, fit_preds),
            "MAE": mae(Y, fit_preds),
        }
        fscore = fit_scores["R2"]
    elif method.endswith("C") and len(set(Y)) == 2:
        fit_scores = {
            "stat": "fit",
            "ROC_AUC": roc_auc_score(Y, fit_preds),
            "ACC": accuracy_score(Y, fit_preds),
            "BAC": balanced_accuracy_score(Y, fit_preds),
            "F1": f1_score(Y, fit_preds),
            "MCC": matthews_corrcoef(Y, fit_preds),
        }
        fscore = fit_scores["BAC"]
    elif method.endswith("C") and len(set(Y)) > 2:
        fit_scores = {
            "stat": "fit",
            "ROC_AUC": roc_auc_score(
                LabelBinarizer().fit_transform(Y),
                LabelBinarizer().fit_transform(fit_preds),
                multi_class="ovr",
            ),
            "ACC": accuracy_score(Y, fit_preds),
            "BAC": balanced_accuracy_score(Y, fit_preds),
            "F1": f1_score(Y, fit_preds, average="macro"),
            "MCC": matthews_corrcoef(Y, fit_preds),
        }
        fscore = fit_scores["BAC"]

    storage[n] = {
        "fit_score": fscore,
        "desc": desc,
        "scaling": scaling,
        "method": method,
        **params,
    }

    score_df = pd.concat([score_df, pd.DataFrame([fit_scores])], ignore_index=True)

    if write_output:
        score_df.to_csv(
            os.path.join(outdir, "trial." + str(n), "stats"),
            sep=" ",
            float_format="%.3f",
            index=False,
        )
        res_pd.to_csv(
            os.path.join(outdir, "trial." + str(n), "predictions"),
            sep=" ",
            float_format="%.3f",
            index=False,
        )
        with open(
            os.path.join(outdir, "trial." + str(n), "parameters.json"), "w"
        ) as param_file:
            param_output = copy.deepcopy(storage[n])
            param_output["ntrials"] = ntrials
            param_output["cv_splits"] = cv_splits
            param_output["cv_repeats"] = cv_repeats
            param_output["jobs"] = jobs
            param_output["timeout"] = tmout
            param_output["earlystop"] = earlystop
            json.dump(param_output, param_file, indent=4)
    else:
        results_detailed[n] = {"score": score_df, "predictions": res_pd}

    if method.endswith("R"):
        score = np.mean(score_df[score_df["stat"].str.contains("consensus")].R2)
    elif method.endswith("C"):
        score = np.mean(score_df[score_df["stat"].str.contains("consensus")].BAC)
    return score


def run_objective_study_with_timeout(
    storage: MutableMapping[int, Dict[str, Any]],
    results_detailed: MutableMapping[int, Dict[str, Any]],
    x_dict: Dict[str, Any],
    y: DataFrame,
    outdir: str,
    method: str,
    ntrials: int,
    cv_splits: int,
    cv_repeats: int,
    jobs: int,
    tmout: int,
    earlystop: Tuple[int, int],
    write_output: bool,
    trial: optuna.trial.Trial,
) -> float:
    timeouted_objective = timeout_decorator.timeout(
        tmout, timeout_exception=optuna.TrialPruned, use_signals=False
    )(objective_study)
    return timeouted_objective(
        storage,
        results_detailed,
        trial,
        x_dict,
        y,
        outdir,
        method,
        ntrials,
        cv_splits,
        cv_repeats,
        jobs,
        tmout,
        earlystop,
        write_output,
    )


def launch_study(
    x_dict: Dict[str, Any],
    y: DataFrame,
    outdir: str,
    method: str,
    ntrials: int,
    cv_splits: int,
    cv_repeats: int,
    jobs: int,
    tmout: int,
    earlystop: Tuple[int, int],
    write_output: bool = True,
) -> Optional[Tuple[DataFrame, Dict[int, Any]]]:
    ctx = mp.get_context()
    with ctx.Manager() as manager:
        results_dict: MutableMapping[int, Dict[str, Any]] = manager.dict()
        results_detailed: MutableMapping[int, Dict[str, Any]] = manager.dict()

        study = optuna.create_study(
            direction="maximize", sampler=optuna.samplers.TPESampler()
        )
        kwargs_opt = (
            {"callbacks": [TopNPatienceCallback(earlystop[0], earlystop[1])]}
            if earlystop[0] > 0
            else {}
        )
        study.optimize(
            partial(
                run_objective_study_with_timeout,
                results_dict,
                results_detailed,
                x_dict,
                y,
                outdir,
                method,
                ntrials,
                cv_splits,
                cv_repeats,
                jobs,
                tmout,
                earlystop,
                write_output,
            ),
            n_trials=ntrials,
            n_jobs=jobs,
            catch=(TimeoutError,),
            **kwargs_opt
        )

        results_dict_local = dict(results_dict)
        results_detailed_local = dict(results_detailed)

    hyperparam_names = list(results_dict_local[next(iter(results_dict_local))].keys())

    results_pd = pd.DataFrame(columns=["trial"] + hyperparam_names + ["score"])
    intermediate = study.trials_dataframe(attrs=("number", "value"))

    for i, row in intermediate.iterrows():
        number = int(row.number)
        if number not in results_dict_local:
            continue
        added_row = {
            "trial": number,
            "score": row.value,
            "fit_score": results_dict_local[number]["fit_score"],
        }
        for hp in hyperparam_names:
            added_row[hp] = results_dict_local[number][hp]

        results_pd = pd.concat(
            [pd.DataFrame([added_row]), results_pd.loc[:]]
        ).reset_index(drop=True)

    if write_output:
        results_pd.to_csv(os.path.join(outdir, "trials.all"), sep=" ", index=False)
        results_pd.sort_values(by="score", ascending=False).head(50).to_csv(
            os.path.join(outdir, "trials.best"), sep=" ", index=False
        )
    else:
        return results_pd, results_detailed_local
    return None


__all__ = ["calculate_scores", "collect_data", "launch_study"]
