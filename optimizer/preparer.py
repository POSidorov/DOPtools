from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from chython import smiles, CGRContainer, MoleculeContainer, from_rdkit_molecule, to_rdkit_molecule
from CGRtools import smiles as cgrtools_smiles
from mordred import Calculator, descriptors
import sys
import multiprocessing
from threading import Thread
import pickle

from cheminfotools.chem_features import ChythonCircus, ChythonLinear, Fingerprinter, ComplexFragmentor, Mordred2DCalculator
from cheminfotools.solvents import SolventVectorizer

import argparse, os
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file
from .config import calculators

parser = argparse.ArgumentParser(prog='Descriptor calculator', 
                                description='Prepares the descriptor files for hyperparameter optimization launch')
parser.add_argument('-i', '--input', required=True, 
                    help='input file, requires csv or Excel format')
parser.add_argument('--structure_col', required=True, action='extend', type=str, nargs='+', default=[])
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

parser.add_argument('--mordred2d', action='store_true', 
                    help='put the option to calculate Mordred 2D descriptors')

parser.add_argument('--solvent', type=str, action='store', default='',
                    help='column that contains the solvents. Check the available solvents in the solvents.py script')

parser.add_argument('--output_structures', action='store_true',
                    help='output the csv file contatining structures along with descriptors')

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
                param_dict[_make_name(('circus',lower, upper))] = {'lower':lower, 'upper':upper}
    if args.linear:
        for lower in _set_default(args.linear_min, [2]):
            for upper in _set_default(args.linear_max, [5]):
                param_dict[_make_name(('chyline',lower, upper))] = {'lower':lower, 'upper':upper}
    if args.mordred2d:
        param_dict[_make_name(('mordred2d',))] = {}
    return param_dict

def _pickle_descriptors(output_dir, fragmentor, prop_name, desc_name):
    fragmentor_name = os.path.join(output_dir, '.'.join([prop_name, desc_name, 'pkl']))
    with open(fragmentor_name, 'wb') as f:
        pickle.dump(fragmentor, f, pickle.HIGHEST_PROTOCOL)

def create_output_dir(outdir):
    if os.path.exists(outdir):
        print('The output directory {} already exists. The data may be overwritten'.format(outdir))
    else:
        os.makedirs(outdir)
        print('The output directory {} created'.format(outdir))

def calculate_descriptors(structures, property_col, desc_name, descriptor_dictionary, output_params):
    # creation of the y vector. if some values are absent, only rows with values will be used
    y = property_col
    indices = list(y[pd.notnull(y)].index)
    if len(indices) < len(property_col):
        print(f"'{p}' column warning: only {len(indices)} out of {len(y)} instances have the property.")
        print(f"Molecules that don't have the property will be discarded from the set.")
        y = y.iloc[indices]
    y = np.array(y)

    desc_type = desc_name.split('_')[0]
    descriptor_params = descriptor_dictionary[desc_name]
    calculator = eval(calculators[desc_type])

    desc = calculator.fit_transform(structures[indices])

    if output_params['write_output']:
        output_folder = output_params['output']
        if output_params['separate']:
            output_folder = os.path.join(output_folder, desc_type)
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print('The output directory {} created'.format(output_folder))

        if output_params['pickle']:
            _pickle_descriptors(output_folder, calculator, property_col.name, desc_name)

        output_name = os.path.join(output_folder, '.'.join([property_col.name, desc_name, output_params['format']]))
        if output_params['format'] == 'csv':
            desc = pd.concat([property_col.iloc[indices], desc], axis=1, sort=False)
            desc.to_csv(output_name, index=False)
        else:
            dump_svmlight_file(np.array(desc), y, output_name, zero_based=False)
    else:
        return desc


if __name__ == '__main__':
    args = parser.parse_args()
    
    output_params = {
        'output': args.output,
        'separate':args.separate_folders,
        'format':args.format,
        'pickle':args.save,
        'write_output': True,
    }
    create_output_dir(output_params['output'])
    
    if args.input.endswith('csv'):
        data_table = pd.read_table(args.input, sep=',')
    elif args.input.endswith('xls') or args.input.endswith('xlsx'):
        data_table = pd.read_excel(args.input)
    else:
        raise ValueError('The input file should be of CSV or Excel format')

    for i, p in enumerate(args.property_col):
        if ' ' in p and len(args.property_names)<i+1:
            raise ValueError('Column names contain spaces or not all alternative names are present.\nPlease provide alternative names with --property_names option')

    solv = None
    if args.solvent:
        sv = SolventVectorizer()

        try:
            solv = sv.transform(data_table[args.solvent])
        except:
            print('Error with the solvent column occurred, solvent descriptors will not be calculated')
            sys.exit()

    structure_dict = {}
    for s in args.structure_col:
        structure_dict[s] = [smiles(m) for m in data_table[s]]

        # this is magic, gives an error if done otherwise...
        for m in structure_dict[s]:
            try:
                m.canonicalize(fix_tautomers=False) 
            except:
                m.canonicalize(fix_tautomers=False)

    structures = [smiles(m) for m in data_table[args.structure_col[0]]]

    # this is magic, gives an error if done otherwise...
    for m in structures:
        try:
            m.canonicalize(fix_tautomers=False) 
        except:
            m.canonicalize(fix_tautomers=False)
    structures = np.array(structures)

    prop = data_table[args.property_col[0]]
    prop.name = args.property_names[0]

    descriptor_dictionary = _enumerate_parameters(args)

    threads = []

    for desc in descriptor_dictionary.keys():
        t = Thread(target=calculate_descriptors, args=(structures, 
                                                    prop, 
                                                    desc,
                                                    descriptor_dictionary,
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
