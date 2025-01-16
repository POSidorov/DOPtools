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
import contextlib
import glob
import os
import warnings
import json
import copy
from functools import partial
from multiprocessing import Manager

import numpy as np
import optuna
import pandas as pd
from optuna.study import StudyDirection
from sklearn.datasets import load_svmlight_file
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import KFold, cross_val_predict
#from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, LabelEncoder, MinMaxScaler
from timeout_decorator import TimeoutError
from timeout_decorator import timeout_decorator
from sklearn.utils import shuffle

from doptools.optimizer.config import get_raw_model, suggest_params
from doptools.optimizer.utils import r2, rmse
from doptools.optimizer.optimizer import *

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

optuna.logging.set_verbosity(optuna.logging.WARNING)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Optuna optimizer',
        description='Optimizes the hyperparameters of ML method on given data, as well as selects the "best" descriptor space.')

    parser.add_argument('-d', '--datadir', required=True,
                        help='Path to the directory containing the descriptors files to run the optimisation on.')
    parser.add_argument('-o', '--outdir', required=True,
                        help='Path to the output directory where the results optimization will be saved.')
    
    parser.add_argument('--ntrials', type=int, default=100,
                        help='Number of hyperparameter sets to explore. After exploring this number of sets, the optimization stops. Default = 100.')
    parser.add_argument('--cv_splits', type=int, default=5,
                        help='Number of folds for K-fold cross-validation. Default = 5.')
    parser.add_argument('--cv_repeats', type=int, default=1,
                        help='Number of times the cross-validation will be repeated with shuffling. Scores are reported as consensus between repeats. Default = 1.')
    
    parser.add_argument('--earlystop_patience', type=int, default=0,
                        help='Number of optimization steps that the best N solutions must not change for the early stopping. By default early stopping is not triggered.')
    parser.add_argument('--earlystop_leaders', type=int, default=1,
                        help='Number N of best solutions that will be checked for the early stopping. Default = 1.')
    parser.add_argument('--timeout', type=int, default=60,
                        help='Timeout in sec. If a trial takes longer it will be killed. Default = 60.')
    
    parser.add_argument('-j', '--jobs', type=int, default=1,
                        help='Number of processes that will be launched in parallel during the optimization. Default = 1.')
    parser.add_argument('-m', '--method', type=str, default='SVR', choices=['SVR', 'SVC', 'RFR', 'RFC', 'XGBR', 'XGBC'],
                        help='ML algorithm to be used for optimization. Only one can be used at a time.')
    #parser.add_argument('--multi', action='store_true')
    parser.add_argument('-f', '--format', type=str, default='svm', choices=['svm', 'csv'],
                        help='Format of the input descriptor files. Default = svm.')
    
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