from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from chython import smiles, CGRContainer, MoleculeContainer, from_rdkit_molecule, to_rdkit_molecule
from CGRtools import smiles as cgrtools_smiles
from mordred import Calculator, descriptors

from cheminfotools.chem_features import ChythonCircus, ChythonIsida, Fingerprinter, ComplexFragmentor

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

parser.add_argument('--morgan', action='store_true', 
                    help='put the option to calculate Morgan fingerprints')
parser.add_argument('--morgan_nBits', type=int, action='store', default=1024, 
                    help='number of bits for Morgan fingerprints')
parser.add_argument('--morgan_radius', nargs='+', action='extend', type=int, default=[2],
                    help='maximum radius of Morgan FP. Allows several numbers, which will be stored separately')

parser.add_argument('--morganfeatures', action='store_true', 
                    help='put the option to calculate Morgan feature fingerprints')
parser.add_argument('--morganfeatures_nBits', type=int, action='store', default=1024, 
                    help='number of bits for Morgan feature fingerprints')
parser.add_argument('--morganfeatures_radius', nargs='+', action='extend', type=int, default=[2],
                    help='maximum radius of Morgan feature FP. Allows several numbers, which will be stored separately')

parser.add_argument('--rdkfp', action='store_true', 
                    help='put the option to calculate RDkit fingerprints')
parser.add_argument('--rdkfp_nBits', type=int, action='store', default=1024, 
                    help='number of bits for RDkit fingerprints')
parser.add_argument('--rdkfp_length', nargs='+', action='extend', type=int, default=[3],
                    help='maximum length of RDkit FP. Allows several numbers, which will be stored separately')

parser.add_argument('--rdkfplinear', action='store_true', 
                    help='put the option to calculate RDkit linear fingerprints')
parser.add_argument('--rdkfplinear_nBits', type=int, action='store', default=1024, 
                    help='number of bits for RDkit linear fingerprints')
parser.add_argument('--rdkfplinear_length', nargs='+', action='extend', type=int, default=[3],
                    help='maximum length of RDkit linear FP. Allows several numbers, which will be stored separately')

parser.add_argument('--layered', action='store_true', 
                    help='put the option to calculate RDkit layered fingerprints')
parser.add_argument('--layered_nBits', type=int, action='store', default=1024, 
                    help='number of bits for RDkit layered fingerprints')
parser.add_argument('--layered_length', nargs='+', action='extend', type=int, default=[3],
                    help='maximum length of RDkit layered FP. Allows several numbers, which will be stored separately')

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
parser.add_argument('--circus_min', nargs='+', action='extend', type=int, default=[0],
                    help='minimum length of ISIDA linear fragments. Allows several numbers, which will be stored separately')
parser.add_argument('--circus_max', nargs='+', action='extend', type=int, default=[2],
                    help='maximum length of ISIDA linear fragments. Allows several numbers, which will be stored separately')

parser.add_argument('--mordred2d', action='store_true', 
                    help='put the option to calculate Mordred 2D descriptors')

def create_output_dir(outdir):
    if os.path.exists(outdir):
        print('The output directory {} already exists. The data may be overwritten'.format(outdir))
    else:
        os.makedirs(outdir)
        print('The output directory {} created'.format(outdir))

def output_file(desc, prop, desctype, outdir, prop_ind_name, fmt='svm', descparams=None, indices=None):
    if fmt not in ['svm', 'csv']:
        raise ValueError('The output file should be of CSV or SVM format')
    outname = outdir + '/'
    try:
        outname += args.property_names[prop_ind_name[0]] +'.'
    except:
        outname += prop_ind_name[1] +'.'
    outname += desctype
    if descparams is not None:
        if desctype == 'isida':
            outname += '-' + str(descparams[0])+'-'+str(descparams[1])+'-'+str(descparams[2])
        elif desctype == 'circus':
            outname += '-' + str(descparams[0])+'-'+str(descparams[1])
        else:
            outname += str(descparams)
    outname += '.'+fmt

    if fmt == 'csv':
        desc = pd.concat([prop.iloc[indices], pd.DataFrame(desc).iloc[indices]], axis=1, sort=False)
        desc.to_csv(outname, index=False)
    else:
        dump_svmlight_file(np.array(desc)[indices], prop.iloc[indices], 
                outname,zero_based=False)


if __name__ == '__main__':
    args = parser.parse_args()
    
    if args.input.endswith('csv'):
        data_table = pd.read_table(args.input, sep=',')
    elif args.input.endswith('xls') or args.input.endswith('xlsx'):
        data_table = pd.read_excel(args.input)
    else:
        raise ValueError('The input file should be of CSV or Excel format')

    for i, p in enumerate(args.property_col):
        if ' ' in p and len(args.property_names)<i+1:
            raise ValueError('Column names contain spaces or not all alternative names are present.\nPlease provide alternative names with --property_names option')

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
        for r in set(args.morgan_radius):
            if len(structure_dict)==1:
                frag = Fingerprinter(fp_type='morgan', nBits=args.morgan_nBits, size=r)
                desc = frag.fit_transform(structure_dict[args.structure_col[0]])
            else:
                frag = ComplexFragmentor(associator=dict(zip([list(structure_dict.keys())],
                                                            [Fingerprinter(fp_type='morgan', 
                                                                nBits=args.morgan_nBits, size=r)]*len(structure_dict.keys()))))
                desc = frag.fit_transform(pd.DataFrame(structure_dict))
            for i, p in enumerate(args.property_col):
                indices = data_table[p][pd.notnull(data_table[p])].index
                if len(indices) < len(data_table[p]):
                    print(f"'{p}' column warning: only {len(indices)} out of {len(data_table[p])} instances have the property.")
                    print(f"Molecules that don't have the property will be discarded from the set.")
                
                output_file(desc, data_table[p], 'morgan', outdir, (i, p), fmt=args.format, descparams=r, indices=indices)
                

    if args.morganfeatures:
        print('Creating a folder for Morgan feature fingerprints')
        outdir = args.output+'/morganfeatures_'+str(args.morgan_nBits)
        create_output_dir(outdir)
        for r in set(args.morganfeatures_radius):
            if len(structure_dict)==1:
                frag = Fingerprinter(fp_type='morgan', nBits=args.morganfeatures_nBits, size=r, 
                                    params={'useFeatures':True})
                desc = frag.fit_transform(structure_dict[args.structure_col[0]])
            else:
                frag = ComplexFragmentor(associator=dict(zip([list(structure_dict.keys())],
                                                            [Fingerprinter(fp_type='morgan', 
                                                                nBits=args.morganfeatures_nBits, size=r,
                                                                params={'useFeatures':True})]*len(structure_dict.keys()))))
                desc = frag.fit_transform(pd.DataFrame(structure_dict))
            for i, p in enumerate(args.property_col):
                indices = data_table[p][pd.notnull(data_table[p])].index
                if len(indices) < len(data_table[p]):
                    print(f"'{p}' column warning: only {len(indices)} out of {len(data_table[p])} instances have the property.")
                    print(f"Molecules that don't have the property will be discarded from the set.")
                
                output_file(desc, data_table[p], 'morganfeatures', outdir, (i, p), fmt=args.format, descparams=r, indices=indices)
                

    if args.rdkfp:
        print('Creating a folder for RDkit fingerprints')
        outdir = args.output+'/rdkfp_'+str(args.rdkfp_nBits)
        create_output_dir(outdir)
        for r in set(args.rdkfp_length):
            if len(structure_dict)==1:
                frag = Fingerprinter(fp_type='rdkfp', nBits=args.rdkfp_nBits, size=r)
                desc = frag.fit_transform(structure_dict[args.structure_col[0]])
            else:
                frag = ComplexFragmentor(associator=dict(zip([list(structure_dict.keys())],
                                                            [Fingerprinter(fp_type='rdkfp', nBits=args.rdkfp_nBits, 
                                                                size=r)]*len(structure_dict.keys()))))
                desc = frag.fit_transform(pd.DataFrame(structure_dict))
            for i, p in enumerate(args.property_col):
                indices = data_table[p][pd.notnull(data_table[p])].index
                if len(indices) < len(data_table[p]):
                    print(f"'{p}' column warning: only {len(indices)} out of {len(data_table[p])} instances have the property.")
                    print(f"Molecules that don't have the property will be discarded from the set.")
                
                output_file(desc, data_table[p], 'rdkfp', outdir, (i, p), fmt=args.format, descparams=r, indices=indices)
                

    if args.rdkfplinear:
        print('Creating a folder for RDkit linear fingerprints')
        outdir = args.output+'/rdkfplinear_'+str(args.rdkfplinear_nBits)
        create_output_dir(outdir)
        for r in set(args.rdkfplinear_length):
            if len(structure_dict)==1:
                frag = Fingerprinter(fp_type='rdkfp', nBits=args.rdkfplinear_nBits, size=r, params={'branchedPaths':False})
                desc = frag.fit_transform(structure_dict[args.structure_col[0]])
            else:
                frag = ComplexFragmentor(associator=dict(zip([list(structure_dict.keys())],
                                                            [Fingerprinter(fp_type='rdkfp', nBits=args.rdkfplinear_nBits, 
                                                                size=r, params={'branchedPaths':False})]*len(structure_dict.keys()))))
                desc = frag.fit_transform(pd.DataFrame(structure_dict))
            for i, p in enumerate(args.property_col):
                indices = data_table[p][pd.notnull(data_table[p])].index
                if len(indices) < len(data_table[p]):
                    print(f"'{p}' column warning: only {len(indices)} out of {len(data_table[p])} instances have the property.")
                    print(f"Molecules that don't have the property will be discarded from the set.")
                
                output_file(desc, data_table[p], 'rdkfplinear', outdir, (i, p), fmt=args.format, descparams=r, indices=indices)
                

    if args.layered:
        print('Creating a folder for RDkit property-layered fingerprints')
        outdir = args.output+'/layered_'+str(args.layered_nBits)
        create_output_dir(outdir)
        for r in set(args.layered_length):
            if len(structure_dict)==1:
                frag = Fingerprinter(fp_type='layered', nBits=args.layered_nBits, size=r)
                desc = frag.fit_transform(structure_dict[args.structure_col[0]])
            else:
                frag = ComplexFragmentor(associator=dict(zip([list(structure_dict.keys())],
                                                            [Fingerprinter(fp_type='rdkfp', nBits=args.layered_nBits, 
                                                                size=r)]*len(structure_dict.keys()))))
                desc = frag.fit_transform(pd.DataFrame(structure_dict))
            for i, p in enumerate(args.property_col):
                indices = data_table[p][pd.notnull(data_table[p])].index
                if len(indices) < len(data_table[p]):
                    print(f"'{p}' column warning: only {len(indices)} out of {len(data_table[p])} instances have the property.")
                    print(f"Molecules that don't have the property will be discarded from the set.")
                
                output_file(desc, data_table[p], 'layered', outdir, (i, p), fmt=args.format, descparams=r, indices=indices)
                 

    if args.avalon:
        print('Creating a folder for Avalon fingerprints')
        outdir = args.output+'/avalon_'+str(args.avalon_nBits)
        create_output_dir(outdir)
        if len(structure_dict)==1:
            frag = Fingerprinter(fp_type='avalon', nBits=args.avalon_nBits)
            desc = frag.fit_transform(structure_dict[args.structure_col[0]])
        else:
            frag = ComplexFragmentor(associator=dict(zip([list(structure_dict.keys())],
                                                            [Fingerprinter(fp_type='avalon', 
                                                                nBits=args.avalon_nBits)]*len(structure_dict.keys()))))
            desc = frag.fit_transform(pd.DataFrame(structure_dict))
        for i, p in enumerate(args.property_col):
            indices = data_table[p][pd.notnull(data_table[p])].index
            if len(indices) < len(data_table[p]):
                print(f"'{p}' column warning: only {len(indices)} out of {len(data_table[p])} instances have the property.")
                print(f"Molecules that don't have the property will be discarded from the set.")
                
            output_file(desc, data_table[p], 'avalon', outdir, (i, p), fmt=args.format, descparams=None, indices=indices)
                  

    if args.atompairs:
        print('Creating a folder for atom pair fingerprints')
        outdir = args.output+'/atompairs_'+str(args.atompairs_nBits)
        create_output_dir(outdir)
        if len(structure_dict)==1:
            frag = Fingerprinter(fp_type='ap', nBits=args.atompairs_nBits)
            desc = frag.fit_transform(structure_dict[args.structure_col[0]])
        else:
            frag = ComplexFragmentor(associator=dict(zip([list(structure_dict.keys())],
                                                            [Fingerprinter(fp_type='ap', 
                                                                nBits=args.atompairs_nBits)]*len(structure_dict.keys()))))
            desc = frag.fit_transform(pd.DataFrame(structure_dict))
        for i, p in enumerate(args.property_col):
            indices = data_table[p][pd.notnull(data_table[p])].index
            if len(indices) < len(data_table[p]):
                print(f"'{p}' column warning: only {len(indices)} out of {len(data_table[p])} instances have the property.")
                print(f"Molecules that don't have the property will be discarded from the set.")
                
            output_file(desc, data_table[p], 'atompairs', outdir, (i, p), fmt=args.format, descparams=None, indices=indices)
                

    if args.torsion:
        print('Creating a folder for topological torsion fingerprints')
        outdir = args.output+'/avalon_'+str(args.torsion_nBits)
        create_output_dir(outdir)
        if len(structure_dict)==1:
            frag = Fingerprinter(fp_type='torsion', nBits=args.torsion_nBits)
            desc = frag.fit_transform(structure_dict[args.structure_col[0]])
        else:
            frag = ComplexFragmentor(associator=dict(zip([list(structure_dict.keys())],
                                                            [Fingerprinter(fp_type='torsion', 
                                                                nBits=args.torsion_nBits)]*len(structure_dict.keys()))))
        desc = frag.fit_transform(pd.DataFrame(structure_dict))
        for i, p in enumerate(args.property_col):
            indices = data_table[p][pd.notnull(data_table[p])].index
            if len(indices) < len(data_table[p]):
                print(f"'{p}' column warning: only {len(indices)} out of {len(data_table[p])} instances have the property.")
                print(f"Molecules that don't have the property will be discarded from the set.")
                
            output_file(desc, data_table[p], 'torsion', outdir, (i, p), fmt=args.format, descparams=None, indices=indices)
                 

    if args.isida:
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
                        
                    output_file(desc, data_table[p], 'isida', outdir, (i, p), fmt=args.format, 
                        descparams=(3, l, u), indices=indices)
                       
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
                        
                    output_file(desc, data_table[p], 'isida', outdir, (i, p), fmt=args.format, 
                        descparams=(9, l, u), indices=indices)

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
                        
                    output_file(desc, data_table[p], 'isida', outdir, (i, p), fmt=args.format, 
                        descparams=(6, l, u), indices=indices)

    if args.circus:
        print('Creating a folder for CircuS fragments')
        outdir = args.output+'/circus'
        create_output_dir(outdir)
        for l in set(args.circus_min):
            for u in set(args.circus_max):
                if len(structure_dict)==1:
                    frag = ChythonCircus(lower=l, upper=u)
                    desc = frag.fit_transform(structure_dict[args.structure_col[0]])
                else:
                    frag = ComplexFragmentor(associator=dict(zip([list(structure_dict.keys())],
                                                                [ChythonCircus(lower=l, 
                                                                    upper=u)]*len(structure_dict.keys()))))
                    desc = frag.fit_transform(pd.DataFrame(structure_dict))
                for i, p in enumerate(args.property_col):
                    indices = data_table[p][pd.notnull(data_table[p])].index
                    if len(indices) < len(data_table[p]):
                        print(f"'{p}' column warning: only {len(indices)} out of {len(data_table[p])} instances have the property.")
                        print(f"Molecules that don't have the property will be discarded from the set.")
                        
                    output_file(desc, data_table[p], 'circus', outdir, (i, p), fmt=args.format, 
                        descparams=(l, u), indices=indices)

    if args.mordred2d:
        print('Creating a folder for Mordred 2D fragments')
        outdir = args.output+'/mordred2D'
        create_output_dir(outdir)
        if len(structure_dict)==1:
            calc = Calculator(descriptors, ignore_3D=True)
            mols = [Chem.MolFromSmiles(str(m)) for m in structure_dict[args.structure_col[0]]]
            desc = calc.pandas(mols).select_dtypes(include='number')
        else:
            descs = []
            for k, v in structure_dict.items():
                calc = Calculator(descriptors, ignore_3D=True)
                mols = [Chem.MolFromSmiles(str(m)) for m in v]
                descs.append(calc.pandas(mols).select_dtypes(include='number'))
            desc = pd.concat(descs, axis=1, sort=False)
        for i, p in enumerate(args.property_col):
            indices = data_table[p][pd.notnull(data_table[p])].index
            if len(indices) < len(data_table[p]):
                print(f"'{p}' column warning: only {len(indices)} out of {len(data_table[p])} instances have the property.")
                print(f"Molecules that don't have the property will be discarded from the set.")
                        
            output_file(desc, data_table[p], 'mordred2d', outdir, (i, p), fmt=args.format, 
                descparams=None, indices=indices)