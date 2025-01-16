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
import multiprocessing as mp
import json

import numpy as np
import pandas as pd
from chython import smiles
from sklearn.datasets import dump_svmlight_file

from doptools.chem.chem_features import ComplexFragmentor, PassThrough
from doptools.chem.solvents import SolventVectorizer
from doptools.optimizer.config import get_raw_calculator
from doptools.optimizer.preparer import *

basic_params = {
    "circus": True, 
    "circus_min": [0], 
    "circus_max": [2, 3, 4], 
    "linear": True, 
    "linear_min": [2], 
    "linear_max": [5, 6, 7, 8], 
    "morgan": True, 
    "morgan_nBits": [1024], 
    "morgan_radius": [2, 3, 4], 
    "morganfeatures": True, 
    "morganfeatures_nBits": [1024], 
    "morganfeatures_radius": [2, 3, 4], 
    "rdkfp": True, 
    "rdkfp_nBits": [1024], 
    "rdkfp_length": [2,3,4], 
    "rdkfplinear": True, 
    "rdkfplinear_nBits": [1024], 
    "rdkfplinear_length": [5,6,7,8], 
    "layered": True, 
    "layered_nBits": [1024], 
    "layered_length": [5,6,7,8], 
    "avalon": True, 
    "avalon_nBits": [1024], 
    "atompairs": True, 
    "atompairs_nBits": [1024], 
    "torsion": True, 
    "torsion_nBits": [1024], 
    "separate_folders": True,
    "save":True,
    "standardize": True
}

def _set_default(argument, default_values):
    if len(argument) > 0:
        return list(set(argument))
    else:
        return default_values


def _enumerate_parameters(args):
    def _make_name(iterable):
        return '_'.join([str(i) for i in iterable])

    param_dict = {}
    if args.morgan:
        for nb in _set_default(args.morgan_nBits, [1024]):
            for mr in _set_default(args.morgan_radius, [2]):
                param_dict[_make_name(('morgan', nb, mr))] = {'nBits': nb, 'radius': mr}
    if args.morganfeatures:
        for nb in _set_default(args.morganfeatures_nBits, [1024]):
            for mr in _set_default(args.morganfeatures_radius, [2]):
                param_dict[_make_name(('morganfeatures', nb, mr))] = {'nBits': nb, 'radius': mr}
    if args.rdkfp:
        for nb in _set_default(args.rdkfp_nBits, [1024]):
            for rl in _set_default(args.rdkfp_length, [3]):
                param_dict[_make_name(('rdkfp', nb, rl))] = {'nBits': nb, 'radius': rl}
    if args.rdkfplinear:
        for nb in _set_default(args.rdkfplinear_nBits, [1024]):
            for rl in _set_default(args.rdkfplinear_length, [3]):
                param_dict[_make_name(('rdkfplinear', nb, rl))] = {'nBits': nb, 'radius': rl}
    if args.layered:
        for nb in _set_default(args.layered_nBits, [1024]):
            for rl in _set_default(args.layered_length, [3]):
                param_dict[_make_name(('layered', nb, rl))] = {'nBits': nb, 'radius': rl}
    if args.avalon:
        for nb in _set_default(args.avalon_nBits, [1024]):
            param_dict[_make_name(('avalon', nb))] = {'nBits': nb}
    if args.torsion:
        for nb in _set_default(args.torsion_nBits, [1024]):
            param_dict[_make_name(('torsion', nb))] = {'nBits': nb}
    if args.atompairs:
        for nb in _set_default(args.atompairs_nBits, [1024]):
            param_dict[_make_name(('atompairs', nb))] = {'nBits': nb}
    if args.circus:
        for lower in _set_default(args.circus_min, [1]):
            for upper in _set_default(args.circus_max, [2]):
                if int(lower) <= int(upper):
                    if args.onbond:
                        param_dict[_make_name(('circus_b', lower, upper))] = {'lower': lower, 'upper': upper, 'on_bond': True}
                    else:
                        param_dict[_make_name(('circus', lower, upper))] = {'lower': lower, 'upper': upper}
    if args.linear:
        for lower in _set_default(args.linear_min, [2]):
            for upper in _set_default(args.linear_max, [5]):
                if int(lower) <= int(upper):
                    param_dict[_make_name(('chyline', lower, upper))] = {'lower': lower, 'upper': upper}
    if args.mordred2d:
        param_dict[_make_name(('mordred2d',))] = {}
    return param_dict


def _pickle_descriptors(output_dir, fragmentor, prop_name, desc_name):
    fragmentor_name = os.path.join(output_dir, '.'.join([prop_name, desc_name, 'pkl']))
    with open(fragmentor_name, 'wb') as f:
        pickle.dump(fragmentor, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Descriptor calculator', 
                                     description='Prepares the descriptor files for hyperparameter optimization launch.')
    
    # I/O arguments
    parser.add_argument('-i', '--input', required=True, 
                        help='Input file, requires csv or Excel format')
    parser.add_argument('--structure_col', action='store', type=str, default='SMILES',
                        help='Column name with molecular structures representations. Default = SMILES.')
    parser.add_argument('--concatenate', action='extend', type=str, nargs='+', default=[],
                        help='Additional column names with molecular structures representations to be concatenated with the primary structure column.')
    parser.add_argument('--property_col', required=True, action='extend', type=str, nargs='+', default=[],
                        help='Column with properties to be used. Case sensitive.')
    parser.add_argument('--property_names', action='extend', type=str, nargs='+', default=[],
                        help='Alternative name for the property columns specified by --property_col.')
    parser.add_argument('--standardize', action='store_true', default=False,
                        help='Standardize the input structures? Default = False.')
    parser.add_argument('-o', '--output', required=True,
                         help='Output folder where the descriptor files will be saved.')
    parser.add_argument('-f', '--format', action='store', type=str, default='svm', choices=['svm', 'csv'],
                        help='Descriptor files format. Default = svm.')
    parser.add_argument('-p', '--parallel', action='store', type=int, default=0,
                        help='Number of parallel processes to use. Default = 0')
    parser.add_argument('-s', '--save', action='store_true',
                        help='Save (pickle) the fragmentors for each descriptor type.')
    parser.add_argument('--separate_folders', action='store_true',
                        help='Save each descriptor type into a separate folders.')
    parser.add_argument('--load_config', action='store', type=str, default='',
                        help='Load descriptor configuration from a JSON file. JSON parameters are prioritized! Use "basic" to load default parameters')

    # Morgan fingerprints
    parser.add_argument('--morgan', action='store_true', 
                        help='Option to calculate Morgan fingerprints.')
    parser.add_argument('--morgan_nBits', nargs='+', action='extend', type=int, default=[],
                        help='Number of bits for Morgan FP. Allows several numbers, which will be stored separately. Default = 1024.')
    parser.add_argument('--morgan_radius', nargs='+', action='extend', type=int, default=[],
                        help='Maximum radius of Morgan FP. Allows several numbers, which will be stored separately. Default = 2.')

    # Morgan feature fingerprints
    parser.add_argument('--morganfeatures', action='store_true', 
                        help='Option to calculate Morgan feature fingerprints.')
    parser.add_argument('--morganfeatures_nBits', nargs='+', action='extend', type=int, default=[],
                        help='Number of bits for Morgan feature FP. Allows several numbers, which will be stored separately. Default = 1024.')
    parser.add_argument('--morganfeatures_radius', nargs='+', action='extend', type=int, default=[],
                        help='Maximum radius of Morgan feature FP. Allows several numbers, which will be stored separately. Default = 2.')

    # RDKit fingerprints
    parser.add_argument('--rdkfp', action='store_true', 
                        help='Option to calculate RDkit fingerprints.')
    parser.add_argument('--rdkfp_nBits', nargs='+', action='extend', type=int, default=[],
                        help='Number of bits for RDkit FP. Allows several numbers, which will be stored separately. Default = 1024.')
    parser.add_argument('--rdkfp_length', nargs='+', action='extend', type=int, default=[],
                        help='Maximum length of RDkit FP. Allows several numbers, which will be stored separately. Default = 3.')

    # RDKit linear fingerprints
    parser.add_argument('--rdkfplinear', action='store_true', 
                        help='Option to calculate RDkit linear fingerprints.')
    parser.add_argument('--rdkfplinear_nBits', nargs='+', action='extend', type=int, default=[],
                        help='Number of bits for RDkit linear FP. Allows several numbers, which will be stored separately. Default = 1024.')
    parser.add_argument('--rdkfplinear_length', nargs='+', action='extend', type=int, default=[],
                        help='Maximum length of RDkit linear FP. Allows several numbers, which will be stored separately. Default = 3.')

    # RDKit layered fingerprints
    parser.add_argument('--layered', action='store_true', 
                        help='Option to calculate RDkit layered fingerprints.')
    parser.add_argument('--layered_nBits', nargs='+', action='extend', type=int, default=[],
                        help='Number of bits for RDkit layered FP. Allows several numbers, which will be stored separately. Default = 1024.')
    parser.add_argument('--layered_length', nargs='+', action='extend', type=int, default=[],
                        help='Maximum length of RDkit layered FP. Allows several numbers, which will be stored separately. Default = 3.')

    # Avalon fingerprints
    parser.add_argument('--avalon', action='store_true', 
                        help='Option to calculate Avalon fingerprints.')
    parser.add_argument('--avalon_nBits', nargs='+', action='extend', type=int, default=[], 
                        help='Number of bits for Avalon FP. Allows several numbers, which will be stored separately. Default = 1024.')

    # Atom pair fingerprints
    parser.add_argument('--atompairs', action='store_true', 
                        help='Option to calculate atom pair fingerprints.')
    parser.add_argument('--atompairs_nBits', nargs='+', action='extend', type=int, default=[],
                        help='Number of bits for atom pair FP. Allows several numbers, which will be stored separately. Default = 1024.')

    # Topological torsion fingerprints
    parser.add_argument('--torsion', action='store_true', 
                        help='Option to calculate topological torsion fingerprints.')
    parser.add_argument('--torsion_nBits', nargs='+', action='extend', type=int, default=[], 
                        help='Number of bits for topological torsion FP. Allows several numbers, which will be stored separately. Default = 1024.')

    # Chython Linear fragments
    parser.add_argument('--linear', action='store_true', 
                        help='Option to calculate ChyLine fragments.')
    parser.add_argument('--linear_min', nargs='+', action='extend', type=int, default=[],
                        help='Minimum length of linear fragments. Allows several numbers, which will be stored separately. Default = 2.')
    parser.add_argument('--linear_max', nargs='+', action='extend', type=int, default=[],
                        help='Maximum length of linear fragments. Allows several numbers, which will be stored separately. Default = 5.')

    # CircuS (Circular Substructures) fragments
    parser.add_argument('--circus', action='store_true', 
                        help='Option to calculate CircuS fragments.')
    parser.add_argument('--circus_min', nargs='+', action='extend', type=int, default=[],
                        help='Minimum radius of CircuS fragments. Allows several numbers, which will be stored separately. Default = 1.')
    parser.add_argument('--circus_max', nargs='+', action='extend', type=int, default=[],
                        help='Maximum radius of CircuS fragments. Allows several numbers, which will be stored separately. Default = 2.')
    parser.add_argument('--onbond', action='store_true', 
                        help='Toggle the calculation of CircuS fragments on bonds. With this option the fragments will be bond-cetered, making a bond the minimal element.')

    # Mordred 2D fingerprints
    parser.add_argument('--mordred2d', action='store_true', 
                        help='Option to calculate Mordred 2D descriptors.')
    # Solvents
    parser.add_argument('--solvent', type=str, action='store', default='',
                        help='Column that contains the solvents. Check the available solvents in the solvents.py script.')


    args = parser.parse_args()

    if args.load_config == "basic":
        vars(args).update(basic_params)
    elif args.load_config:
        with open(args.load_config) as f:
            p = json.load(f)
        vars(args).update(p)

    check_parameters(args)
    
    input_params = {
        'input_file': args.input,
        'structure_col': args.structure_col,
        'standardize': args.standardize,
        'property_col': args.property_col,
        'property_names': args.property_names,
        'concatenate': args.concatenate,
        'solvent': args.solvent
    }

    output_params = {
        'output': args.output,
        'separate': args.separate_folders,
        'format': args.format,
        'pickle': args.save,
        'write_output': True,
    }
    create_output_dir(output_params['output'])

    inpt = create_input(input_params)

    descriptor_dictionary = _enumerate_parameters(args)

    # Create a multiprocessing pool (excluding mordred) with the specified number of processes
    # If args.parallel is 0 or negative, use the default number of processes
    pool = mp.Pool(processes=args.parallel if args.parallel > 0 else 1)
    non_mordred_descriptors = [desc for desc in descriptor_dictionary.keys() if 'mordred2d' not in desc]
    # Use pool.map to apply the calculate_and_output function to each set of arguments in parallel
    # The arguments are tuples containing (inpt, descriptor, descriptor_params, output_params)
    pool.map(calculate_and_output, [(inpt, desc, descriptor_dictionary[desc], output_params) for desc in non_mordred_descriptors])
    pool.close() # Close the pool and prevent any more tasks from being submitted
    pool.join() # Wait for all the tasks to complete

    # Serial mordred calculations
    mordred_descriptors = [desc for desc in descriptor_dictionary.keys() if 'mordred2d' in desc]
    for desc in mordred_descriptors:
        calculate_and_output((inpt, desc, descriptor_dictionary[desc], output_params))
