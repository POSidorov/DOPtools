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

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)


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


def check_parameters(params):
    if params.input.split('.')[-1] not in ('csv', 'xls', 'xlsx'):
        raise ValueError('The input file should be of CSV or Excel format.')
    for i, p in enumerate(params.property_col):
        if ' ' in p and len(params.property_names)<(i+1):
            raise ValueError(f'Column name {p} contains spaces in the name.\nPlease provide alternative names with --property_names option.')
    if params.property_names:
        if len(params.property_col) != len(params.property_names):
            raise ValueError('The number of alternative names is not equal to the number of properties.')
        

def create_input(input_params):
    input_dict = {}
    structures = []

    if input_params['input_file'].endswith('csv'):
        data_table = pd.read_table(input_params['input_file'], sep=',')
    elif input_params['input_file'].endswith('xls') or input_params['input_file'].endswith('xlsx'):
        data_table = pd.read_excel(input_params['input_file'])
    else:
        raise ValueError("Input file format not supported. Please use csv, xls or xlsx.")

    input_dict['structures'] = pd.DataFrame(columns=[input_params['structure_col']] + input_params['concatenate'])
    for col in [input_params['structure_col']] + input_params['concatenate']:
        structures = [smiles(m) for m in data_table[col]]
        if input_params['standardize']:
            # this is magic, gives an error if done otherwise...
            for m in structures:
                try:
                    m.canonicalize(fix_tautomers=False) 
                except:
                    m.canonicalize(fix_tautomers=False)
        input_dict['structures'][col] = structures
    #input_dict['structures'] = structures

    if input_params['solvent']:
        input_dict['solvents'] = data_table[input_params['solvent']]

    if 'passthrough' in input_params.keys() and input_params['passthrough']:
        input_dict['passthrough'] = data_table[list(input_params['passthrough'])]

    for i, p in enumerate(input_params['property_col']):
        y = data_table[p]
        indices = list(y[pd.notnull(y)].index)
        if len(indices) < len(structures):
            print(f"'{p}' column warning: only {len(indices)} out of {len(structures)} instances have the property.")
            print(f"Molecules that don't have the property will be discarded from the set.")
            y = y.iloc[indices]
        y = np.array(y)

        if input_params['property_names']:
            name = input_params['property_names'][i]
        else:
            name = p

        input_dict['prop'+str(i+1)] = {'indices': indices,
                                       'property': y,
                                       'property_name': name}
    return input_dict


def calculate_descriptor_table(input_dict, desc_name, descriptor_params, out='all'):
    desc_type = desc_name.split('_')[0]
    result = {'name': desc_name, 'type': desc_type}
    for k, d in input_dict.items():
        if k.startswith('prop'):
            base_column = list(input_dict['structures'].columns)[0]
            if len(input_dict['structures'].columns) == 1 and 'solvents' not in input_dict.keys() \
                    and 'passthrough' not in input_dict.keys():
                calculator = get_raw_calculator(desc_type, descriptor_params)
                desc = calculator.fit_transform(input_dict['structures'][base_column].iloc[d['indices']])
            else:
                calculators_dict = {}
                for c in input_dict['structures'].columns:
                    calculators_dict[c] = get_raw_calculator(desc_type, descriptor_params)
                input_table = input_dict['structures']
                if 'solvents' in input_dict.keys():
                    calculators_dict[input_dict['solvents'].name] = SolventVectorizer()
                    input_table = pd.concat([input_dict['structures'], input_dict['solvents']], axis=1)
                if 'passthrough' in input_dict.keys():
                    if type(input_dict['passthrough']) is not pd.DataFrame:
                        input_dict['passthrough'] = pd.DataFrame(input_dict['passthrough'])
                    for pt in input_dict['passthrough']:
                        calculators_dict[pt] = PassThrough(column_name=pt)
                        input_table = pd.concat([input_table, input_dict['passthrough'][pt]], axis=1)

                calculator = ComplexFragmentor(associator=calculators_dict,
                                               structure_columns=[base_column])
                desc = calculator.fit_transform(input_table.iloc[d['indices']])

            result[k] = {'calculator': calculator, 'table': desc, 
                         'name': d['property_name'], 'property': d['property']}

    if out == 'all':
        return result
    elif out in list(result.keys()):
        return result[out]
    else:
        raise ValueError('The return value is not in the result dictionary')


def output_descriptors(calculated_result, output_params):
    desc_name = calculated_result['name']
    desc_type = calculated_result['type']

    output_folder = output_params['output']
    if output_params['separate']:
        output_folder = os.path.join(output_folder, desc_type)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)  # exist_ok is useful when several processes try to create the folder at the same time
        print('The output directory {} created'.format(output_folder))
    for k, d in calculated_result.items():
        if k.startswith('prop'):
            if output_params['pickle']:
                _pickle_descriptors(output_folder, d['calculator'], 
                                    d['name'], desc_name)

            output_name = os.path.join(output_folder, '.'.join([d['name'], 
                                                                desc_name, 
                                                                output_params['format']]))
            if output_params['format'] == 'csv':
                desc = pd.concat([pd.Series(d['property'], name=d['name']), d['table']], axis=1, sort=False)
                desc.to_csv(output_name, index=False)
            else:
                dump_svmlight_file(np.array(d['table']), d['property'], 
                                   output_name, zero_based=False)


def calculate_and_output(input_args):
    inpt, desc, descriptor_params, output_params = input_args
    result = calculate_descriptor_table(inpt, desc, descriptor_params)
    output_descriptors(result, output_params)


def create_output_dir(outdir):
    if os.path.exists(outdir):
        print('The output directory {} already exists. The data may be overwritten'.format(outdir))
    else:
        os.makedirs(outdir)
        print('The output directory {} created'.format(outdir))


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
                        help='Load descriptor configuration from a JSON file. JSON parameters are prioritized!')

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
    if args.load_config:
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


__all__ = ['calculate_and_output', 'calculate_descriptor_table', 'check_parameters',
           'create_input', 'create_output_dir', 'output_descriptors']
