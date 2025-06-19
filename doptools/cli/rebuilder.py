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

import os
import pandas as pd
import pickle
import argparse
import glob

from typing import Optional, List, Dict, Tuple, Iterable

from doptools.optimizer.config import get_raw_model

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import VarianceThreshold

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


class Rebuilder:
    def __init__(self, file:str=None, 
                       folders:List[str]=None, 
                       ensemble:int=0):
        self.file = file
        self.folders = folders
        if self.file is None and self.folders is None:
            raise ValueError("At least one file or folder should be given to rebuild models")
        self.ensemble = ensemble


    def gather_trials(self):
        trial_files = []
        if self.folders is not None:
            for f in self.folders:
                trial_files.append(glob.glob(os.path.join(f, 'trials.all')))
        elif self.file is not None:
            trial_files.append(self.file)

        full_df = pd.concat([pd.read_table(f, sep=" ") for f in trial_files])
        return full_df



def rebuild_from_file(descdir, modeldir, number):
    trials = pd.read_table(os.path.join(modeldir, 'trials.all'), sep=' ')
    rebuild_trial = trials[trials['trial'] == number].squeeze()

    trial_preds = pd.read_table(os.path.join(modeldir, 'trial.'+str(number), 'predictions'), sep=' ')
    prop = list(trial_preds.columns)[1].removesuffix('.observed')

    pipeline_steps = []

    desc_name = rebuild_trial['desc']
    if os.path.isdir(os.path.join(descdir, desc_name.split('-')[0])):
        desc_file = os.path.join(descdir, desc_name.split('-')[0], prop+'.'+desc_name+'.pkl')
    else:
        desc_file = os.path.join(descdir, prop+'.'+desc_name+'.pkl')
    with open(desc_file, 'rb') as f:
        desc_calculator = pickle.load(f)
    pipeline_steps.append(('descriptor_calculator', desc_calculator))

    if rebuild_trial['scaling'] == 'scaled':
        pipeline_steps.append(('scaler', MinMaxScaler()))

    pipeline_steps.append(('variance', VarianceThreshold()))

    params = rebuild_trial[rebuild_trial.index[list(rebuild_trial.index).index('method')+1:]].to_dict()
    for k, p in params.items():
        if pd.isnull(p):
            params[k] = None
        ## The next check is necessary for max_features in RFR and RFC
        if k == "max_features":
            try:
                params[k] = float(params[k])
            except:
                pass
    method = rebuild_trial['method']
    model = get_raw_model(method, params)
    pipeline_steps.append(('model', model))

    pipeline = Pipeline(pipeline_steps)

    return pipeline, rebuild_trial

def rebuilder():
    parser = argparse.ArgumentParser(prog='Optimized model rebuilder', 
                                     description='Rebuilds the model from the optimized trial parameters,\nsaving it as an UNTRAINED pipeline in pickle')
    parser.add_argument('-d', '--descdir', required=True,
                        help='the folder containing descriptor files and calculators. Can contain folders separated by descriptor type')
    parser.add_argument('-m', '--modeldir', required=True,
                        help='the folder containing model output files. Should contain "trials.all" file.')
    parser.add_argument('-n', '--number', type=int, required=True,
                        help='the trial number for the model to be rebuilt.')
    parser.add_argument('-o', '--outdir', required=True,
                        help='the output folder for the models.')

    args = parser.parse_args()
    descdir = args.descdir
    modeldir = args.modeldir
    number = args.number
    outdir = args.outdir

    if os.path.exists(outdir):
        print('The output directory {} already exists. The data may be overwritten'.format(outdir))
    else:
        os.makedirs(outdir)
        print('The output directory {} created'.format(outdir))

    pipeline, trial = rebuild_from_file(descdir, modeldir, number)

    modelfile_name = '_'.join([trial['method'], 'trial'+str(number), trial['desc']])
    with open(os.path.join(outdir, modelfile_name+'.pkl'), 'wb') as f:
        pickle.dump(pipeline, f)


__all__ = ['rebuild_from_file']
