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

from cheminfotools.chem_features import ChythonCircus, Fingerprinter, ComplexFragmentor
try:
    from cheminfotools.fragmentor import ChythonIsida
    isida_able = True
except:
    print('ISIDA Fragmentor could not be loaded. Check the installation')
    isida_able = False
from cheminfotools.solvents import SolventVectorizer

import argparse, os
import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file, dump_svmlight_file

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

parser.add_argument('--morgan', action='store_true', 
                    help='put the option to calculate Morgan fingerprints')
parser.add_argument('--morgan_nBits', type=int, action='store', default=1024, 
                    help='number of bits for Morgan fingerprints')
parser.add_argument('--morgan_radius', nargs='+', action='extend', type=int, default=[],
                    help='maximum radius of Morgan FP. Allows several numbers, which will be stored separately. Default radius 2')

parser.add_argument('--morganfeatures', action='store_true', 
                    help='put the option to calculate Morgan feature fingerprints')
parser.add_argument('--morganfeatures_nBits', type=int, action='store', default=1024, 
                    help='number of bits for Morgan feature fingerprints')
parser.add_argument('--morganfeatures_radius', nargs='+', action='extend', type=int, default=[],
                    help='maximum radius of Morgan feature FP. Allows several numbers, which will be stored separately. Default radius 2')

parser.add_argument('--rdkfp', action='store_true', 
                    help='put the option to calculate RDkit fingerprints')
parser.add_argument('--rdkfp_nBits', type=int, action='store', default=1024, 
                    help='number of bits for RDkit fingerprints')
parser.add_argument('--rdkfp_length', nargs='+', action='extend', type=int, default=[],
                    help='maximum length of RDkit FP. Allows several numbers, which will be stored separately. Default length 3')

parser.add_argument('--rdkfplinear', action='store_true', 
                    help='put the option to calculate RDkit linear fingerprints')
parser.add_argument('--rdkfplinear_nBits', type=int, action='store', default=1024, 
                    help='number of bits for RDkit linear fingerprints')
parser.add_argument('--rdkfplinear_length', nargs='+', action='extend', type=int, default=[],
                    help='maximum length of RDkit linear FP. Allows several numbers, which will be stored separately. Default length 3')

parser.add_argument('--layered', action='store_true', 
                    help='put the option to calculate RDkit layered fingerprints')
parser.add_argument('--layered_nBits', type=int, action='store', default=1024, 
                    help='number of bits for RDkit layered fingerprints')
parser.add_argument('--layered_length', nargs='+', action='extend', type=int, default=[],
                    help='maximum length of RDkit layered FP. Allows several numbers, which will be stored separately. Default length 3')

parser.add_argument('--avalon', action='store_true', 
                    help='put the option to calculate Avalon fingerprints')
parser.add_argument('--avalon_nBits', type=int, action='store', default=1024, 
                    help='number of bits for Avalon fingerprints')

parser.add_argument('--atompairs', action='store_true', 
                    help='put the option to calculate atom pair fingerprints')
parser.add_argument('--atompairs_nBits', type=int, action='store', default=1024, 
                    help='number of bits for atom pair fingerprints')

parser.add_argument('--torsion', action='store_true', 
                    help='put the option to calculate topological torsion fingerprints')
parser.add_argument('--torsion_nBits', type=int, action='store', default=1024, 
                    help='number of bits for topological torsion fingerprints')

parser.add_argument('--linear', action='store_true', 
                    help='put the option to calculate linear fragments')
parser.add_argument('--linear_min', nargs='+', action='extend', type=int, default=[],
                    help='minimum length of linear fragments. Allows several numbers, which will be stored separately. Default value 2')
parser.add_argument('--linear_max', nargs='+', action='extend', type=int, default=[],
                    help='maximum length of linear fragments. Allows several numbers, which will be stored separately. Default value 5')

parser.add_argument('--isida', action='store_true', 
                    help='put the option to calculate ISIDA fragments')
parser.add_argument('--isida_linear_min', nargs='+', action='extend', type=int, default=[2],
                    help='minimum length of ISIDA linear fragments. Allows several numbers, which will be stored separately')
parser.add_argument('--isida_linear_max', nargs='+', action='extend', type=int, default=[5],
                    help='maximum length of ISIDA linear fragments. Allows several numbers, which will be stored separately')
parser.add_argument('--isida_circular_min', nargs='+', action='extend', type=int, default=[1],
                    help='minimum length of ISIDA atom-centered fragments of fixed length. Allows several numbers, which will be stored separately')
parser.add_argument('--isida_circular_max', nargs='+', action='extend', type=int, default=[3],
                    help='maximum length of ISIDA atom-centered fragments of fixed length. Allows several numbers, which will be stored separately')
parser.add_argument('--isida_flex_min', nargs='+', action='extend', type=int, default=[1],
                    help='minimum length of ISIDA atom-centered fragments. Allows several numbers, which will be stored separately')
parser.add_argument('--isida_flex_max', nargs='+', action='extend', type=int, default=[3],
                    help='maximum length of ISIDA atom-centered fragments. Allows several numbers, which will be stored separately')

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
        return set(argument)
    else:
        return default_values

def _pickle_descriptors(output_dir, desc_type, fragmentor, prop_ind_name):
    fragmentor_name = output_dir+'/'
    try:
        fragmentor_name += args.property_names[prop_ind_name[0]] +'.'
    except:
        fragmentor_name += prop_ind_name[1] +'.'
    fragmentor_name += desc_type+'-'
    if desc_type == 'circus' or desc_type == 'linear':
        fragmentor_name += str(fragmentor.lower) + '-' + str(fragmentor.upper)
    else:
        try:
            fragmentor_name += str(fragmentor.size)
        except:
            pass
    fragmentor_name += '.pkl'
    with open(fragmentor_name, 'wb') as f:
        pickle.dump(fragmentor, f, pickle.HIGHEST_PROTOCOL)

def create_output_dir(outdir):
    if os.path.exists(outdir):
        print('The output directory {} already exists. The data may be overwritten'.format(outdir))
    else:
        os.makedirs(outdir)
        print('The output directory {} created'.format(outdir))

def output_file(desc, prop, desc_type, outdir, prop_ind_name, solvent=None, 
                fmt='svm', structures=None, descparams=None, indices=None):
    if fmt not in ['svm', 'csv']:
        raise ValueError('The output file should be of CSV or SVM format')
    outname = outdir + '/'
    try:
        outname += args.property_names[prop_ind_name[0]] +'.'
    except:
        outname += prop_ind_name[1] +'.'
    outname += desc_type
    if descparams is not None:
        if desc_type == 'isida':
            outname += '-' + str(descparams[0])+'-'+str(descparams[1])+'-'+str(descparams[2])
        elif desc_type == 'circus' or desc_type == 'linear':
            outname += '-' + str(descparams[0])+'-'+str(descparams[1])
        else:
            outname += '-'+str(descparams)
    outname += '.'+fmt

    if solvent is not None:
        desc = pd.concat([desc, solvent], axis=1, sort=False)

    if fmt == 'csv':
        if structures is not None:
            struc_col = pd.Series([str(s) for s in structures[indices]], name='SMILES')
            desc = pd.concat([struc_col, prop.iloc[indices], 
                                pd.DataFrame(desc).iloc[indices]], axis=1, sort=False)
        else:
            desc = pd.concat([prop.iloc[indices], pd.DataFrame(desc)], axis=1, sort=False)
        desc.to_csv(outname, index=False)
    else:
        dump_svmlight_file(np.array(desc), prop.iloc[indices], 
                outname,zero_based=False)

def calculate_descriptors(data, structures, properties, desc_type, other_params, output_dir, output_params, save):
    def _create_calculator(dtype, prms):
        if dtype == 'circus':
            return ChythonCircus(lower=prms['lower'], upper=prms['upper'])
        if dtype == 'linear':
            return ChythonLinear(lower=prms['lower'], upper=prms['upper'])
        elif dtype == 'mordred2d':
            return Calculator(descriptors, ignore_3D=True)
        else:
            return Fingerprinter(fp_type=dtype, **prms)

    for i, p in enumerate(properties):
        indices = data[pd.notnull(data[p])].index
        if len(indices) < len(data[p]):
            print(f"'{p}' column warning: only {len(indices)} out of {len(data[p])} instances have the property.")
            print(f"Molecules that don't have the property will be discarded from the set.")
        if len(structures.keys())==1:    
            strs = np.array(structures[next(iter(structures))])[indices]
            frag = _create_calculator(desc_type, other_params)
            if desc_type == 'mordred2d':
                mols = [Chem.MolFromSmiles(str(m)) for m in strs]
                desc = frag.pandas(mols).select_dtypes(include='number')
            else:
                desc = frag.fit_transform(strs)
                if save:
                    _pickle_descriptors(output_dir, desc_type, frag, (i,p))
        else:
            # make a ComplexFragmentor
            strs = dict(zip([(key, np.array(structures[key])[indices]) for key in structures.keys()]))
            if desc_type == 'mordred2d':
                descs = []
                for k, v in structures.items():
                    calc = Calculator(descriptors, ignore_3D=True)
                    mols = [Chem.MolFromSmiles(str(m)) for m in v]
                    descs.append(calc.pandas(mols).select_dtypes(include='number'))
                desc = pd.concat(descs, axis=1, sort=False)
            else:
                frag = ComplexFragmentor(associator=dict(zip([list(structures.keys())],
                                            [_create_calculator(desc_type, other_params)]*len(strs.keys()))))
                desc = frag.fit_transform(pd.DataFrame(strs))
                if save:
                    _pickle_descriptors(output_dir, desc_type, frag, (i,p))

        output_file(desc, data[p], desc_type, output_dir, (i, p), fmt=output_params['format'],
                    solvent=solv, structures=strs, descparams=frag.get_size(), indices=indices)

if __name__ == '__main__':
    args = parser.parse_args()

    threads = []
    
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

    if args.morgan:
        print('Creating a folder for Morgan fingerprints')
        outdir = args.output+'/morgan_'+str(args.morgan_nBits)
        create_output_dir(outdir)
        radii = _set_default(args.morgan_radius, [2])
        for r in radii:
            t = Thread(target=calculate_descriptors, args=(data_table, structure_dict, 
                                                    data_table[args.property_col], 
                                                    'morgan', 
                                                    {'nBits':args.morgan_nBits, 'size':r}, 
                                                    outdir, {'format':args.format},
                                                    args.save))
            threads.append(t)
            
    
    if args.morganfeatures:
        print('Creating a folder for Morgan feature fingerprints')
        outdir = args.output+'/morganfeatures_'+str(args.morgan_nBits)
        create_output_dir(outdir)
        radii = _set_default(args.morganfeatures_radius, [2])
        for r in radii:
            t = Thread(target=calculate_descriptors, args=(data_table, structure_dict, 
                                                    data_table[args.property_col], 
                                                    'morgan', 
                                                    {'nBits':args.morganfeatures_nBits, 'size':r,
                                                      'params':{'useFeatures':True}}, 
                                                    outdir, {'format':args.format},
                                                    args.save))
            threads.append(t)
                

    if args.rdkfp:
        print('Creating a folder for RDkit fingerprints')
        outdir = args.output+'/rdkfp_'+str(args.rdkfp_nBits)
        create_output_dir(outdir)
        radii = _set_default(args.rdfkp_length, [3])
        for r in radii:
            t = Thread(target=calculate_descriptors, args=(data_table, structure_dict, 
                                                    data_table[args.property_col], 
                                                    'rdkfp', 
                                                    {'nBits':args.rdkfp_nBits, 'size':r}, 
                                                    outdir, {'format':args.format},
                                                    args.save))
            threads.append(t)
                

    if args.rdkfplinear:
        print('Creating a folder for RDkit linear fingerprints')
        outdir = args.output+'/rdkfplinear_'+str(args.rdkfplinear_nBits)
        create_output_dir(outdir)
        radii = _set_default(args.rdkfplinear_length, [3])
        for r in radii:
            t = Thread(target=calculate_descriptors, args=(data_table, structure_dict, 
                                                    data_table[args.property_col], 
                                                    'rdkfp', 
                                                    {'nBits':args.rdkfplinear_nBits, 'size':r,
                                                     'params':{'branchedPaths':False}}, 
                                                    outdir, {'format':args.format},
                                                    args.save))
            threads.append(t)
                

    if args.layered:
        print('Creating a folder for RDkit property-layered fingerprints')
        outdir = args.output+'/layered_'+str(args.layered_nBits)
        create_output_dir(outdir)
        radii = _set_default(args.layered_length, [3])
        for r in radii:
            t = Thread(target=calculate_descriptors, args=(data_table, structure_dict, 
                                                    data_table[args.property_col], 
                                                    'layered', 
                                                    {'nBits':args.layered_nBits, 'size':r}, 
                                                    outdir, {'format':args.format},
                                                    args.save))
            threads.append(t)
                 

    if args.avalon:
        print('Creating a folder for Avalon fingerprints')
        outdir = args.output+'/avalon_'+str(args.avalon_nBits)
        create_output_dir(outdir)
        t = Thread(target=calculate_descriptors, args=(data_table, structure_dict, 
                                                    data_table[args.property_col], 
                                                    'avalon', 
                                                    {'nBits':args.avalon_nBits}, 
                                                    outdir, {'format':args.format},
                                                    args.save))
        threads.append(t)
                  

    if args.atompairs:
        print('Creating a folder for atom pair fingerprints')
        outdir = args.output+'/atompairs_'+str(args.atompairs_nBits)
        create_output_dir(outdir)
        t = Thread(target=calculate_descriptors, args=(data_table, structure_dict, 
                                                    data_table[args.property_col], 
                                                    'ap', 
                                                    {'nBits':args.atompairs_nBits}, 
                                                    outdir, {'format':args.format},
                                                    args.save))
        threads.append(t)
                

    if args.torsion:
        print('Creating a folder for topological torsion fingerprints')
        outdir = args.output+'/torsion_'+str(args.torsion_nBits)
        create_output_dir(outdir)
        t = Thread(target=calculate_descriptors, args=(data_table, structure_dict, 
                                                    data_table[args.property_col], 
                                                    'torsion', 
                                                    {'nBits':args.torsion_nBits}, 
                                                    outdir, {'format':args.format},
                                                    args.save))
        threads.append(t)
                        
                 
    if args.isida and isida_able:
        print('Creating a folder for ISIDA fragments')
        outdir = args.output+'/isida'
        create_output_dir(outdir)
        for l in set(args.isida_linear_min):
            for u in set(args.isida_linear_max):
                if len(structure_dict)==1:
                    frag = ChythonIsida(ftype=3, lower=l, upper=u)
                    desc = frag.fit_transform(structure_dict[args.structure_col[0]])
                else:
                    frag = ComplexFragmentor(associator=dict(zip([list(structure_dict.keys())],
                                                                [ChythonIsida(ftype=3, lower=l, 
                                                                    upper=u)]*len(structure_dict.keys()))))
                    desc = frag.fit_transform(pd.DataFrame(structure_dict))
                
                for i, p in enumerate(args.property_col):
                    indices = data_table[p][pd.notnull(data_table[p])].index
                    if len(indices) < len(data_table[p]):
                        print(f"'{p}' column warning: only {len(indices)} out of {len(data_table[p])} instances have the property.")
                        print(f"Molecules that don't have the property will be discarded from the set.")
                    structures = None
                    if args.output_structures:
                        structures = np.array(structure_dict[args.structure_col[0]])[indices]    
                    output_file(desc, data_table[p], 'isida', outdir, (i, p), fmt=args.format, 
                        solvent=solv, structures=structures, descparams=(3, l, u), indices=indices)
                       
        for l in set(args.isida_circular_min):
            for u in set(args.isida_circular_max):
                if len(structure_dict)==1:
                    frag = ChythonIsida(ftype=9, lower=l, upper=u)
                    desc = frag.fit_transform(structure_dict[args.structure_col[0]])
                else:
                    frag = ComplexFragmentor(associator=dict(zip([list(structure_dict.keys())],
                                                                [ChythonIsida(ftype=9, lower=l, 
                                                                    upper=u)]*len(structure_dict.keys()))))
                    desc = frag.fit_transform(pd.DataFrame(structure_dict))
                
                for i, p in enumerate(args.property_col):
                    indices = data_table[p][pd.notnull(data_table[p])].index
                    if len(indices) < len(data_table[p]):
                        print(f"'{p}' column warning: only {len(indices)} out of {len(data_table[p])} instances have the property.")
                        print(f"Molecules that don't have the property will be discarded from the set.")
                    structures = None
                    if args.output_structures:
                        structures = np.array(structure_dict[args.structure_col[0]])[indices]    
                    output_file(desc, data_table[p], 'isida', outdir, (i, p), fmt=args.format, 
                        solvent=solv, structures=structures, descparams=(9, l, u), indices=indices)

        for l in set(args.isida_flex_min):
            for u in set(args.isida_flex_max):
                if len(structure_dict)==1:
                    frag = ChythonIsida(ftype=6, lower=l, upper=u)
                    desc = frag.fit_transform(structure_dict[args.structure_col[0]])
                else:
                    frag = ComplexFragmentor(associator=dict(zip([list(structure_dict.keys())],
                                                                [ChythonIsida(ftype=6, lower=l, 
                                                                    upper=u)]*len(structure_dict.keys()))))
                    desc = frag.fit_transform(pd.DataFrame(structure_dict))
                
                for i, p in enumerate(args.property_col):
                    indices = data_table[p][pd.notnull(data_table[p])].index
                    if len(indices) < len(data_table[p]):
                        print(f"'{p}' column warning: only {len(indices)} out of {len(data_table[p])} instances have the property.")
                        print(f"Molecules that don't have the property will be discarded from the set.")
                    structures = None
                    if args.output_structures:
                        structures = np.array(structure_dict[args.structure_col[0]])[indices]    
                    output_file(desc, data_table[p], 'isida', outdir, (i, p), fmt=args.format, 
                        solvent=solv, structures=structures, descparams=(6, l, u), indices=indices)

    if args.circus:
        print('Creating a folder for CircuS fragments')
        outdir = args.output+'/circus'
        create_output_dir(outdir)
        lowers = _set_default(args.circus_min, [1])
        uppers = _set_default(args.circus_max, [2])
        for l in lowers:
            for u in uppers:
                t = Thread(target=calculate_descriptors, args=(data_table, structure_dict, 
                                                    data_table[args.property_col], 
                                                    'circus', 
                                                    {'lower':l, 'upper':u}, 
                                                    outdir, {'format':args.format},
                                                    args.save))
                threads.append(t)

    if args.linear:
        print('Creating a folder for linear fragments')
        outdir = args.output+'/linear'
        create_output_dir(outdir)
        lowers = _set_default(args.linear_min, [2])
        uppers = _set_default(args.linear_max, [5])
        for l in lowers:
            for u in uppers:
                t = Thread(target=calculate_descriptors, args=(data_table, structure_dict, 
                                                    data_table[args.property_col], 
                                                    'linear', 
                                                    {'lower':l, 'upper':u}, 
                                                    outdir, {'format':args.format},
                                                    args.save))
                threads.append(t)
                

    if args.mordred2d:
        print('Creating a folder for Mordred 2D fragments')
        outdir = args.output+'/mordred2d'
        create_output_dir(outdir)
        t = Thread(target=calculate_descriptors, args=(data_table, structure_dict, 
                                                    data_table[args.property_col], 
                                                    'mordred2d', {}, 
                                                    outdir, {'format':args.format},
                                                    args.save))
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