import optuna
import glob, contextlib, os
import pandas as pd
import numpy as np
import time, os
import timeout_decorator
from timeout_decorator.timeout_decorator import TimeoutError
from multiprocessing import Manager
from functools import partial

from config import suggest_params

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
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

parser = argparse.ArgumentParser(prog='Optuna optimizer', 
                                description='Optimizes the hyperparameters of ML method on given data, similar to Dragos\'s optimizer')
parser.add_argument('-d', '--datadir', required=True)
parser.add_argument('-o', '--outdir', required=True)
parser.add_argument('--ntrials', type=int, default=100)
parser.add_argument('--cv_splits', type=int, default=5)
parser.add_argument('--cv_repeats', type=int, default=1)
parser.add_argument('--timeout', type=int, default=60)
parser.add_argument('-j', '--jobs', type=int, default=1)
parser.add_argument('-m', '--method', type=str, default='SVR', choices=['SVR', 'SVC', 'RFR', 'XGBR'])
parser.add_argument('--multi', action='store_true')
parser.add_argument('-f', '--format', type=str, default='svm', choices=['svm', 'csv'])

optuna.logging.set_verbosity(optuna.logging.WARNING)

def r2(a, b):
    return 1. - np.sum((a-b)**2)/np.sum((a-np.mean(a))**2)

def rmse(a, b):
    return np.sqrt(np.sum((a-b)**2)/len(a))


def collect_data(datadir, multi):
    desc_dict = {}
    y = {}
    for f in glob.glob(datadir+"/*.svm"):
        name = f.split('/')[-1][:-4].split('.')[1]
        propname = f.split('/')[-1].split('.')[0]
        desc_dict[name], y[propname] = load_svmlight_file(f)
    return desc_dict, pd.DataFrame(y)

def launch_study(x_dict, y, outdir, method, ntrials, cv_splits, cv_repeats, jobs, tmout, multi):
    manager = Manager()
    results_dict = manager.dict()

    @timeout_decorator.timeout(tmout, timeout_exception=optuna.TrialPruned, use_signals=False)
    def objective(storage, trial):
        n = trial.number
        if not os.path.exists(outdir+'/trial.'+str(n)):
            os.mkdir(outdir+'/trial.'+str(n))
        res_pd = pd.DataFrame(columns=['data_index'])
        res_pd['data_index'] = np.arange(1, len(y)+1, step=1).astype(int)
        
        desc = trial.suggest_categorical('desc_type', list(x_dict.keys()))
        scaling = trial.suggest_categorical('scaling', ['scaled', 'original'])
        X = x_dict[desc].toarray()
        
        #if set(X.flatten()) == set([0,1]):
        #    scaling = 'original'
        #else:
        #    scaling = 'scaled'
        #    mms = MinMaxScaler()
        #    X = mms.fit_transform(x_dict[desc].toarray())

        if scaling == 'scaled':
            mms = MinMaxScaler()
            X = mms.fit_transform(X)
        #else:
        #    X = x_dict[desc].toarray()

        X = VarianceThreshold().fit_transform(X)

        params = suggest_params(trial, method)
        storage[n] = {'desc': desc, 'scaling': scaling, **params}
        if method == 'SVR':
            model = SVR(**params, gamma='auto')
        if method == 'SVC':
            model = SVC(**params, gamma='auto')
        if method == 'XGBR':
            model = xgb.XGBRegressor(**params, verbosity=0, nthread=1)
        if method == 'XGBC':
            model = xgb.XGBClassifier(**params, verbosity=0, nthread=1)
        if method == 'RFR':
            model = RandomForestRegressor(**params)
        if method == 'RFC':
            model = RandomForestClassifier(**params)
        if method.endswith('R'):
            score_df = pd.DataFrame(columns=['stat', 'R2', 'RMSE', 'MAE'])
        elif method.endswith('C'):
            score_df = pd.DataFrame(columns=['stat', 'ROC_AUC', 'ACC', 'BAC', 'F1'])
        if multi:
            model = MultiOutputRegressor(model)
            Y = y
        else:
            Y = np.array(y[y.columns[0]])

        for r in range(cv_repeats):
            preds = cross_val_predict(model, X, Y, cv=KFold(cv_splits, shuffle=True))
            if len(y.columns)<2:
                preds = preds.reshape((-1, 1))
            for i, c in enumerate(y.columns):
                res_pd[c + '.observed'] = y[c]
                res_pd[c + '.predicted.repeat'+str(r+1)] = preds[:,i]

        for c in y.columns:
            preds_partial = res_pd[[d for d in res_pd.columns if c+'.predicted' in d]]
            for p in preds_partial.columns:
                if method.endswith('R'):
                    added_row = {'stat':p, 'R2':r2(y[c], preds_partial[p]),
                             'RMSE':rmse(y[c], preds_partial[p]), 
                             'MAE':mae(y[c], preds_partial[p])}
                elif method.endswith('C'):
                    added_row = {'stat':p, 'ROC_AUC':roc_auc_acore(y[c], preds_partial[p]),
                             'ACC':accuracy_score(y[c], preds_partial[p]), 
                             'BAC':balanced_accuracy_score(y[c], preds_partial[p]),
                             'F1':f1_score(y[c], preds_partial[p])}
                score_df = pd.concat([pd.DataFrame(added_row, index=[0]), score_df.loc[:]]).reset_index(drop=True)
            if method.endswith('R'):
                added_row = {'stat':c+'.consensus', 'R2':r2(y[c], preds_partial.mean(axis=1)),
                             'RMSE':rmse(y[c], preds_partial.mean(axis=1)), 
                             'MAE':mae(y[c], preds_partial.mean(axis=1))}
            elif method.endswith('C'):
                added_row = {'stat':p, 'ROC_AUC':roc_auc_score(y[c], np.round(preds_partial.mean(axis=1)).astype(int)),
                             'ACC':accuracy_score(y[c], np.round(preds_partial.mean(axis=1)).astype(int)), 
                             'BAC':balanced_accuracy_score(y[c], np.round(preds_partial.mean(axis=1)).astype(int)),
                             'F1':f1_score(y[c], np.round(preds_partial.mean(axis=1)).astype(int))}
            score_df = pd.concat([pd.DataFrame(added_row, index=[0]), score_df.loc[:]]).reset_index(drop=True)

        score_df.to_csv(outdir+'/trial.'+str(n)+'/stats', sep=' ', float_format='%.3f', index=False)
        res_pd.to_csv(outdir+'/trial.'+str(n)+'/predictions', sep=' ', float_format='%.3f', index=False)  

        if method.endswith('R'):
            score = np.mean(score_df[score_df['stat'].str.contains('consensus')].R2)
        elif method.endswith('C'):
            score = np.mean(score_df[score_df['stat'].str.contains('consensus')].BAC)
        return score

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(partial(objective, results_dict), n_trials=ntrials, n_jobs=jobs, catch=(TimeoutError,))
    
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
    
    results_pd.to_csv(outdir+'/trials.all', sep=' ', index=False) 
    if ntrials>50:
        results_pd.sort_values(by='score', ascending=False).head(50).to_csv(outdir+'/trials.best', sep=' ', index=False) 
    else:
        results_pd.sort_values(by='score', ascending=False).to_csv(outdir+'/trials.best', sep=' ', index=False) 


if __name__ == '__main__':
    args = parser.parse_args()
    datadir = args.datadir
    outdir = args.outdir
    ntrials = args.ntrials
    cv_splits = args.cv_splits
    cv_repeats = args.cv_repeats
    tmout = args.timeout
    jobs = args.jobs
    method = args.method
    multi = args.multi

    if os.path.exists(outdir):
        print('The output directory {} already exists. The data may be overwritten'.format(outdir))
    else:
        os.makedirs(outdir)
        print('The output directory {} created'.format(outdir))

    x_dict, y = collect_data(datadir, multi)
    
    with contextlib.redirect_stdout(open(os.devnull, "w")):
        launch_study(x_dict, y, outdir, method, ntrials, cv_splits, cv_repeats, jobs, tmout, multi)
