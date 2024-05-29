#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
#  Copyright 2022-2024 Pavel Sidorov <pavel.o.sidorov@gmail.com> This
#  file is part of DOPTools repository.
#
#  ChemInfoTools is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful, but
#  WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
#  General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see
#  <https://www.gnu.org/licenses/>.

import optuna
from optuna.study import StudyDirection
import glob, contextlib, os
import pandas as pd
import numpy as np
import timeout_decorator
from timeout_decorator.timeout_decorator import TimeoutError
from multiprocessing import Manager
from functools import partial

from doptools.optimizer.config import suggest_params, methods

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import RepeatedKFold, cross_val_score, KFold, cross_val_predict
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVR, SVC
import xgboost as xgb
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from sklearn.metrics import mean_absolute_error as mae 
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score
from sklearn.multioutput import MultiOutputRegressor

import argparse

optuna.logging.set_verbosity(optuna.logging.WARNING)

def r2(a, b):
    return 1. - np.sum((a-b)**2)/np.sum((a-np.mean(a))**2)

def rmse(a, b):
    return np.sqrt(np.sum((a-b)**2)/len(a))

class TopNPatienceCallback:
    def __init__(self, patience: int, leaders:int = 1):
        self.patience = patience
        self.leaders = leaders
        self._leaders_unchanged_steps = 0
        self._previous_leaders = ()

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if len(study.trials_dataframe()) < self.leaders:
            return

        if study.direction == StudyDirection.MAXIMIZE:
            new_leaders = tuple(study.trials_dataframe().sort_values(by='value', ascending=False).head(self.leaders).number)
        else:
            new_leaders = tuple(study.trials_dataframe().sort_values(by='value', ascending=True).head(self.leaders).number)

        if new_leaders == self._previous_leaders:
            self._leaders_unchanged_steps += 1
        else:
            self._leaders_unchanged_steps = 0
            self._previous_leaders = new_leaders

        if self._leaders_unchanged_steps >= self.patience:
            study.stop()

def collect_data(datadir, task, fmt='svm'):
    desc_dict = {}
    y = {}
    for f in glob.glob(datadir+"/*."+fmt):
        name = f.split('/')[-1][:-4].split('.')[1]
        propname = f.split('/')[-1].split('.')[0]
        fullname = f.split('/')[-1]
        if fmt == 'svm':
            desc_dict[name], y[propname] = load_svmlight_file(f)
        elif fmt == 'csv':
            data = pd.read_table(f)
            y[propname] = data[propname]
            col_idx = list(data.columns).index()
            desc_dict[name] = data.iloc[:,col_idx+1:]
    if task.endswith('C'):
        return desc_dict, pd.DataFrame(y, dtype=int)
    else:
        return desc_dict, pd.DataFrame(y)

def calculate_scores(task, obs, pred):
    def create_row(task, stat_name, x, y):
        if task == 'R':
            return {'stat':stat_name, 'R2':r2(x, y), 'RMSE':rmse(x, y), 'MAE':mae(x, y)}
        elif task == 'C' and len(set(x))==2:
            return {'stat':stat_name, 'ROC_AUC':roc_auc_score(x, y), 'ACC':accuracy_score(x, y), 
                             'BAC':balanced_accuracy_score(x, y), 'F1':f1_score(x, y)}
        elif task == 'C' and len(set(x))>2:
            return {'stat':stat_name, 'ROC_AUC':roc_auc_score(LabelBinarizer().fit_transform(x), 
                                                LabelBinarizer().fit_transform(y), multi_class='ovr'), 
                    'ACC':accuracy_score(x, y), 
                    'BAC':balanced_accuracy_score(x, y), 
                    'F1':f1_score(x, y, average='macro')}

    if task == 'R':
        score_df = pd.DataFrame(columns=['stat', 'R2', 'RMSE', 'MAE'])
    elif task == 'C':
        score_df = pd.DataFrame(columns=['stat', 'ROC_AUC', 'ACC', 'BAC', 'F1'])

    for c in obs.columns:
        preds_partial = pred[[d for d in pred.columns if c+'.predicted' in d]]
        for p in preds_partial.columns:
            added_row = create_row(task, p, obs[c], preds_partial[p])
            score_df = pd.concat([pd.DataFrame(added_row, index=[0]), score_df.loc[:]]).reset_index(drop=True)
        if task == 'R':
            added_row = create_row(task, c+'.consensus', obs[c], preds_partial.mean(axis=1))
        elif task == 'C':
            added_row = create_row(task, c+'.consensus', obs[c], np.round(preds_partial.mean(axis=1)))
        score_df = pd.concat([pd.DataFrame(added_row, index=[0]), score_df.loc[:]]).reset_index(drop=True)
    return score_df


def launch_study(x_dict, y, outdir, method, ntrials, cv_splits, cv_repeats, jobs, tmout, earlystop, write_output: bool = True):
    manager = Manager()
    results_dict = manager.dict()
    results_detailed = manager.dict()

    @timeout_decorator.timeout(tmout, timeout_exception=optuna.TrialPruned, use_signals=False)
    def objective(storage, results_detailed, trial):
        n = trial.number
        if write_output and not os.path.exists(outdir+'/trial.'+str(n)):
            os.mkdir(outdir+'/trial.'+str(n))
        res_pd = pd.DataFrame(columns=['data_index'])
        res_pd['data_index'] = np.arange(1, len(y)+1, step=1).astype(int)
        
        desc = trial.suggest_categorical('desc_type', list(x_dict.keys()))
        scaling = trial.suggest_categorical('scaling', ['scaled', 'original'])
        X = x_dict[desc].toarray()

        if scaling == 'scaled':
            mms = MinMaxScaler()
            X = mms.fit_transform(X)

        X = VarianceThreshold().fit_transform(X)

        params = suggest_params(trial, method)
        storage[n] = {'desc': desc, 'scaling': scaling, 'method':method, **params}

        model = eval(methods[method])

        #if multi:
        #    model = MultiOutputRegressor(model)
        #    Y = y
        #else:
        Y = np.array(y[y.columns[0]])
        if method.endswith('C'):
            LE = LabelEncoder()
            Y = LE.fit_transform(Y)

        for r in range(cv_repeats):
            preds = cross_val_predict(model, X, Y, cv=KFold(cv_splits, shuffle=True))
            #if len(y.columns)<2:
            if method.endswith('C'):
                preds = LE.inverse_transform(preds)

            preds = preds.reshape((-1, 1))
            for i, c in enumerate(y.columns):
                res_pd[c + '.observed'] = y[c]
                res_pd[c + '.predicted.repeat'+str(r+1)] = preds[:,i]

        score_df = calculate_scores(method[-1], y, res_pd)

        if write_output:
            score_df.to_csv(outdir+'/trial.'+str(n)+'/stats', sep=' ', float_format='%.3f', index=False)
            res_pd.to_csv(outdir+'/trial.'+str(n)+'/predictions', sep=' ', float_format='%.3f', index=False)
        else:
            results_detailed[n] = {'score': score_df, 'predictions': res_pd}

        if method.endswith('R'):
            score = np.mean(score_df[score_df['stat'].str.contains('consensus')].R2)
        elif method.endswith('C'):
            score = np.mean(score_df[score_df['stat'].str.contains('consensus')].BAC)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    if earlystop[0]>0:
        study.optimize(partial(objective, results_dict, results_detailed), n_trials=ntrials, n_jobs=jobs, catch=(TimeoutError,),
                       callbacks=[TopNPatienceCallback(earlystop[0], earlystop[1])])
    else:
        study.optimize(partial(objective, results_dict, results_detailed), n_trials=ntrials, n_jobs=jobs, catch=(TimeoutError,))
    
    hyperparam_names = list(results_dict[next(iter(results_dict))].keys())

    results_pd = pd.DataFrame(columns=['trial']+hyperparam_names+['score'])
    intermediate = study.trials_dataframe(attrs=('number','value'))
    
    for i, row in intermediate.iterrows():
        number = int(row.number)
        if number not in results_dict:
            continue
        
        added_row = {'trial':number,'score':row.value}
        for hp in hyperparam_names:
            added_row[hp] = results_dict[number][hp]
        results_pd = pd.concat([pd.DataFrame(added_row, index=[0]), results_pd.loc[:]]).reset_index(drop=True)
    
    if write_output:
        results_pd.to_csv(outdir+'/trials.all', sep=' ', index=False)
        if ntrials>50:
            results_pd.sort_values(by='score', ascending=False).head(50).to_csv(outdir+'/trials.best', sep=' ', index=False)
        else:
            results_pd.sort_values(by='score', ascending=False).to_csv(outdir+'/trials.best', sep=' ', index=False)
    else:
        return results_pd, results_detailed


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Optuna optimizer', 
                                description='Optimizes the hyperparameters of ML method on given data, similar to Dragos\'s optimizer')
    parser.add_argument('-d', '--datadir', required=True, 
        help='the folder containing descriptor files to run the optimization on.')
    parser.add_argument('-o', '--outdir', required=True, 
        help='the output folder for the results of optimization.')
    parser.add_argument('--ntrials', type=int, default=100, 
        help='number of hyperparameter sets to explore. After exploring this number of sets, the optimization stops.')
    parser.add_argument('--cv_splits', type=int, default=5, 
        help='number of folds for K-fold cross-validation')
    parser.add_argument('--cv_repeats', type=int, default=1, 
        help='number of times the cross-validation will be repeated with shuffling. Scores are reported as consensus between repeats.')
    parser.add_argument('--earlystop_patience', type=int, default=0, 
        help='number of optimization steps that the best N solutions must not change for the early stopping. By default early stopping is not triggered.')
    parser.add_argument('--earlystop_leaders', type=int, default=1, 
        help='number N of best solutions that will be checked for the early stopping.')
    parser.add_argument('--timeout', type=int, default=60, 
        help='timeout in sec. If a trial takes longer it will be killed.')
    parser.add_argument('-j', '--jobs', type=int, default=1, 
        help='number of processes that will be launched in parallel during the optimization.')
    parser.add_argument('-m', '--method', type=str, default='SVR', choices=['SVR', 'SVC', 'RFR', 'RFC', 'XGBR', 'XGBC'], 
        help='ML algorithm to be used for optimization. Only one can be used at a time.')
    #parser.add_argument('--multi', action='store_true')
    parser.add_argument('-f', '--format', type=str, default='svm', choices=['svm', 'csv'])
    args = parser.parse_args()
    datadir = args.datadir
    outdir = args.outdir
    ntrials = args.ntrials
    cv_splits = args.cv_splits
    cv_repeats = args.cv_repeats
    tmout = args.timeout
    jobs = args.jobs
    method = args.method
    #multi = args.multi
    fmt = args.format
    earlystop = (args.earlystop_patience, args.earlystop_leaders)

    if os.path.exists(outdir):
        print('The output directory {} already exists. The data may be overwritten'.format(outdir))
    else:
        os.makedirs(outdir)
        print('The output directory {} created'.format(outdir))

    x_dict, y = collect_data(datadir, method, fmt)
    
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        launch_study(x_dict, y, outdir, method, ntrials, 
                    cv_splits, cv_repeats, jobs, tmout, earlystop)
