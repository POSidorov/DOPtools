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

from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from chython import smiles, CGRContainer, MoleculeContainer, from_rdkit_molecule, to_rdkit_molecule
from mordred import Calculator, descriptors
import sys
import multiprocessing
from threading import Thread
import pickle

from doptools.chem.chem_features import ChythonCircus, ChythonLinear, Fingerprinter, ComplexFragmentor, Mordred2DCalculator
from doptools.chem.solvents import SolventVectorizer

import argparse, os
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from doptools.optimizer.config import calculators


def _set_default(argument, default_values):
    if len(argument)>0:
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
                param_dict[_make_name(('morgan',nb,mr))] = {'nBits':nb, 'radius':mr}
    if args.morganfeatures:
        for nb in _set_default(args.morganfeatures_nBits, [1024]):
            for mr in _set_default(args.morganfeatures_radius, [2]):
                param_dict[_make_name(('morganfeatures',nb,mr))] = {'nBits':nb, 'radius':mr}
    if args.rdkfp:
        for nb in _set_default(args.rdkfp_nBits, [1024]):
            for rl in _set_default(args.rdkfp_length, [3]):
                param_dict[_make_name(('rdkfp',nb,rl))] = {'nBits':nb, 'radius':rl}
    if args.rdkfplinear:
        for nb in _set_default(args.rdkfplinear_nBits, [1024]):
            for rl in _set_default(args.rdkfplinear_length, [3]):
                param_dict[_make_name(('rdkfplinear',nb,rl))] = {'nBits':nb, 'radius':rl}
    if args.layered:
        for nb in _set_default(args.layered_nBits, [1024]):
            for rl in _set_default(args.layered_length, [3]):
                param_dict[_make_name(('layered',nb,rl))] = {'nBits':nb, 'radius':rl}
    if args.avalon:
        for nb in _set_default(args.avalon_nBits, [1024]):
            param_dict[_make_name(('avalon',nb))] = {'nBits':nb}
    if args.torsion:
        for nb in _set_default(args.torsion_nBits, [1024]):
            param_dict[_make_name(('torsion',nb))] = {'nBits':nb}
    if args.atompairs:
        for nb in _set_default(args.atompairs_nBits, [1024]):
            param_dict[_make_name(('atompairs',nb))] = {'nBits':nb}
    if args.circus:
        for lower in _set_default(args.circus_min, [1]):
            for upper in _set_default(args.circus_max, [2]):
                if int(lower) <= int(upper):
                    if args.onbond:
                        param_dict[_make_name(('circus_b',lower, upper))] = {'lower':lower, 'upper':upper, 'on_bond':True}
                    else:
                        param_dict[_make_name(('circus',lower, upper))] = {'lower':lower, 'upper':upper}
    if args.linear:
        for lower in _set_default(args.linear_min, [2]):
            for upper in _set_default(args.linear_max, [5]):
                if int(lower) <= int(upper):
                    param_dict[_make_name(('chyline',lower, upper))] = {'lower':lower, 'upper':upper}
    if args.mordred2d:
        param_dict[_make_name(('mordred2d',))] = {}
    return param_dict

def _pickle_descriptors(output_dir, fragmentor, prop_name, desc_name):
    fragmentor_name = os.path.join(output_dir, '.'.join([prop_name, desc_name, 'pkl']))
    with open(fragmentor_name, 'wb') as f:
        pickle.dump(fragmentor, f, pickle.HIGHEST_PROTOCOL)

def check_parameters(params):
    if params.input.split('.')[-1] not in ('csv', 'xls', 'xlsx') :
        raise ValueError('The input file should be of CSV or Excel format.')
    for i, p in enumerate(params.property_col):
        if ' ' in p:
            raise ValueError(f'Column name {p} contains spaces in the name.\nPlease provide alternative names with --property_names option.')
    if params.property_names:
        if len(params.property_col) != len(params.property_names):
            raise ValueError('The number of alternative names is not equal to the number of properties.')
        

def create_input(input_params):
    input_dict = {}

    if input_params['input_file'].endswith('csv'):
        data_table = pd.read_table(input_params['input_file'], sep=',')
    elif input_params['input_file'].endswith('xls') or input_params['input_file'].endswith('xlsx'):
        data_table = pd.read_excel(input_params['input_file'])
    else:
        raise ValueError("Input file format not supported. Please use csv, xls or xlsx.")

    input_dict['structures'] = pd.DataFrame(columns=[input_params['structure_col']] + input_params['concatenate'])
    for col in [input_params['structure_col']] + input_params['concatenate']:
        structures = [smiles(m) for m in data_table[col]]
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

        input_dict['prop'+str(i+1)] = { 'indices': indices,
                                        'property': y,
                                        'property_name': name}
    return input_dict


def calculate_descriptor_table(input_dict, desc_name, descriptor_params, out='all'):
    desc_type = desc_name.split('_')[0]
    result = {'name':desc_name, 'type':desc_type}
    for k, d in input_dict.items():
        if k.startswith('prop'):
            base_column = list(input_dict['structures'].columns)[0]
            if len(input_dict['structures'].columns) == 1 and 'solvents' not in input_dict.keys():
                calculator = eval(calculators[desc_type])
                desc = calculator.fit_transform(input_dict['structures'][base_column].iloc[d['indices']])
            else:
                calculators_dict = {}
                for c in input_dict['structures'].columns:
                    calculators_dict[c] = eval(calculators[desc_type])
                input_table = input_dict['structures']
                if 'solvents' in input_dict.keys():
                    calculators_dict[input_dict['solvents'].name] = SolventVectorizer()
                    input_table = pd.concat([input_dict['structures'], input_dict['solvents']], axis=1)

                calculator = ComplexFragmentor(associator = calculators_dict, 
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

def calculate_and_output(input_dict, desc_name, descriptor_dictionary, output_params):
    result = calculate_descriptor_table(input_dict, desc_name, descriptor_dictionary)
    output_descriptors(result, output_params)

def create_output_dir(outdir):
    if os.path.exists(outdir):
        print('The output directory {} already exists. The data may be overwritten'.format(outdir))
    else:
        os.makedirs(outdir)
        print('The output directory {} created'.format(outdir))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Descriptor calculator', 
                                     description='Prepares the descriptor files for hyperparameter optimization launch')
    parser.add_argument('-i', '--input', required=True, 
                        help='input file, requires csv or Excel format')
    parser.add_argument('--structure_col', required=True, type=str, 
                        help='the name of the column where the structure SMILES are stored')
    parser.add_argument('--concatenate', action='extend', type=str, nargs='+', default=[])
    parser.add_argument('--property_col', required=True, action='extend', type=str, nargs='+', default=[])
    parser.add_argument('--property_names', action='extend', type=str, nargs='+', default=[])

    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-f', '--format', action='store', type=str, default='svm', choices=['svm', 'csv'])
    parser.add_argument('-p', '--parallel', action='store', type=int, default=0)
    parser.add_argument('-s', '--save', action='store_true', help='save the fragmentors for each descriptor type')
    parser.add_argument('--separate_folders', action='store_true', help='save each descriptor type into a separate folders')

    parser.add_argument('--morgan', action='store_true', 
                        help='put the option to calculate Morgan fingerprints')
    parser.add_argument('--morgan_nBits', nargs='+', action='extend', type=int, default=[],
                        help='number of bits for Morgan fingerprints')
    parser.add_argument('--morgan_radius', nargs='+', action='extend', type=int, default=[],
                        help='maximum radius of Morgan FP. Allows several numbers, which will be stored separately. Default radius 2')

    parser.add_argument('--morganfeatures', action='store_true', 
                        help='put the option to calculate Morgan feature fingerprints')
    parser.add_argument('--morganfeatures_nBits', nargs='+', action='extend', type=int, default=[],
                        help='number of bits for Morgan feature fingerprints')
    parser.add_argument('--morganfeatures_radius', nargs='+', action='extend', type=int, default=[],
                        help='maximum radius of Morgan feature FP. Allows several numbers, which will be stored separately. Default radius 2')

    parser.add_argument('--rdkfp', action='store_true', 
                        help='put the option to calculate RDkit fingerprints')
    parser.add_argument('--rdkfp_nBits', nargs='+', action='extend', type=int, default=[],
                        help='number of bits for RDkit fingerprints')
    parser.add_argument('--rdkfp_length', nargs='+', action='extend', type=int, default=[],
                        help='maximum length of RDkit FP. Allows several numbers, which will be stored separately. Default length 3')

    parser.add_argument('--rdkfplinear', action='store_true', 
                        help='put the option to calculate RDkit linear fingerprints')
    parser.add_argument('--rdkfplinear_nBits', nargs='+', action='extend', type=int, default=[],
                        help='number of bits for RDkit linear fingerprints')
    parser.add_argument('--rdkfplinear_length', nargs='+', action='extend', type=int, default=[],
                        help='maximum length of RDkit linear FP. Allows several numbers, which will be stored separately. Default length 3')

    parser.add_argument('--layered', action='store_true', 
                        help='put the option to calculate RDkit layered fingerprints')
    parser.add_argument('--layered_nBits', nargs='+', action='extend', type=int, default=[],
                        help='number of bits for RDkit layered fingerprints')
    parser.add_argument('--layered_length', nargs='+', action='extend', type=int, default=[],
                        help='maximum length of RDkit layered FP. Allows several numbers, which will be stored separately. Default length 3')

    parser.add_argument('--avalon', action='store_true', 
                        help='put the option to calculate Avalon fingerprints')
    parser.add_argument('--avalon_nBits', nargs='+', action='extend', type=int, default=[], 
                        help='number of bits for Avalon fingerprints')

    parser.add_argument('--atompairs', action='store_true', 
                        help='put the option to calculate atom pair fingerprints')
    parser.add_argument('--atompairs_nBits', nargs='+', action='extend', type=int, default=[],
                        help='number of bits for atom pair fingerprints')

    parser.add_argument('--torsion', action='store_true', 
                        help='put the option to calculate topological torsion fingerprints')
    parser.add_argument('--torsion_nBits', nargs='+', action='extend', type=int, default=[], 
                        help='number of bits for topological torsion fingerprints')

    parser.add_argument('--linear', action='store_true', 
                        help='put the option to calculate ChyLine fragments')
    parser.add_argument('--linear_min', nargs='+', action='extend', type=int, default=[],
                        help='minimum length of linear fragments. Allows several numbers, which will be stored separately. Default value 2')
    parser.add_argument('--linear_max', nargs='+', action='extend', type=int, default=[],
                        help='maximum length of linear fragments. Allows several numbers, which will be stored separately. Default value 5')

    parser.add_argument('--circus', action='store_true', 
                        help='put the option to calculate CircuS fragments')
    parser.add_argument('--circus_min', nargs='+', action='extend', type=int, default=[],
                        help='minimum radius of CircuS fragments. Allows several numbers, which will be stored separately. Default value 1')
    parser.add_argument('--circus_max', nargs='+', action='extend', type=int, default=[],
                        help='maximum radius of CircuS fragments. Allows several numbers, which will be stored separately. Default value 2')
    parser.add_argument('--onbond', action='store_true', 
                        help='toggle the calculation of CircuS fragments on bonds')

    parser.add_argument('--mordred2d', action='store_true', 
                        help='put the option to calculate Mordred 2D descriptors')

    parser.add_argument('--solvent', type=str, action='store', default='',
                        help='column that contains the solvents. Check the available solvents in the solvents.py script')

    parser.add_argument('--output_structures', action='store_true',
                        help='output the csv file contatining structures along with descriptors')

    args = parser.parse_args()
    check_parameters(args)
    
    input_params = {
        'input_file':args.input, 
        'structure_col':args.structure_col,
        'property_col':args.property_col,
        'property_names':args.property_names,
        'concatenate': args.concatenate,
        'solvent':args.solvent
    }

    output_params = {
        'output': args.output,
        'separate':args.separate_folders,
        'format':args.format,
        'pickle':args.save,
        'write_output': True,
    }
    create_output_dir(output_params['output'])

    inpt = create_input(input_params)

    descriptor_dictionary = _enumerate_parameters(args)

    threads = []

    for desc in descriptor_dictionary.keys():
        t = Thread(target=calculate_and_output, args=(inpt, 
                                                      desc,
                                                      descriptor_dictionary[desc],
                                                      output_params))
        threads.append(t)
    
    if args.parallel>0:    
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    else:
        for t in threads:
            t.start()
            t.join()
