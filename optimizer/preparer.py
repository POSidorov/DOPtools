from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon import pyAvalonTools
from chython import smiles, CGRContainer, MoleculeContainer, from_rdkit_molecule, to_rdkit_molecule
from CGRtools import smiles as cgrtools_smiles

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


def create_output_dir(outdir):
    if os.path.exists(outdir):
        print('The output directory {} already exists. The data may be overwritten'.format(outdir))
    else:
        os.makedirs(outdir)
        print('The output directory {} created'.format(outdir))

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
        [m.canonicalize() for m in structure_dict[s]]

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
                try:
                    dump_svmlight_file(desc, data_table[p], 
                        outdir+'/'+args.property_names[i]+".morgan"+str(r)+".svm",zero_based=False)
                except:
                    dump_svmlight_file(desc, data_table[p], 
                        outdir+'/'+p+".morgan"+str(r)+".svm",zero_based=False)

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
                try:
                    dump_svmlight_file(desc, data_table[p], 
                        outdir+'/'+args.property_names[i]+".morganfeatures"+str(r)+".svm",zero_based=False)
                except:
                    dump_svmlight_file(desc, data_table[p], 
                        outdir+'/'+p+".morganfeatures"+str(r)+".svm",zero_based=False)

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
                try:
                    dump_svmlight_file(desc, data_table[p], 
                        outdir+'/'+args.property_names[i]+".rdkfp"+str(r)+".svm",zero_based=False)
                except:
                    dump_svmlight_file(desc, data_table[p], 
                        outdir+'/'+p+".rdkfp"+str(r)+".svm",zero_based=False)

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
                try:
                    dump_svmlight_file(desc, data_table[p], 
                        outdir+'/'+args.property_names[i]+".rdkfplinear"+str(r)+".svm",zero_based=False)
                except:
                    dump_svmlight_file(desc, data_table[p], 
                        outdir+'/'+p+".rdkfplinear"+str(r)+".svm",zero_based=False)   

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
                try:
                    dump_svmlight_file(desc, data_table[p], 
                        outdir+'/'+args.property_names[i]+".layered"+str(r)+".svm",zero_based=False)
                except:
                    dump_svmlight_file(desc, data_table[p], 
                        outdir+'/'+p+".layered"+str(r)+".svm",zero_based=False)  

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
            try:
                dump_svmlight_file(desc, data_table[p], 
                    outdir+'/'+args.property_names[i]+".avalon.svm",zero_based=False)
            except:
                dump_svmlight_file(desc, data_table[p], 
                    outdir+'/'+p+".avalon.svm",zero_based=False)    

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
            try:
                dump_svmlight_file(desc, data_table[p], 
                    outdir+'/'+args.property_names[i]+".atompairs.svm",zero_based=False)
            except:
                dump_svmlight_file(desc, data_table[p], 
                    outdir+'/'+p+".atompairs.svm",zero_based=False)  

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
            try:
                dump_svmlight_file(desc, data_table[p], 
                    outdir+'/'+args.property_names[i]+".torsion.svm",zero_based=False)
            except:
                dump_svmlight_file(desc, data_table[p], 
                    outdir+'/'+p+".torsion.svm",zero_based=False)     

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
                    try:
                        dump_svmlight_file(desc, data_table[p], 
                            outdir+'/'+args.property_names[i]+".isida-3-"+str(l)+"-"+str(u)+".svm",zero_based=False)
                    except:
                        dump_svmlight_file(desc, data_table[p], 
                            outdir+'/'+p+".isida-3-"+str(l)+"-"+str(u)+".svm",zero_based=False)   
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
                    try:
                        dump_svmlight_file(desc, data_table[p], 
                            outdir+'/'+args.property_names[i]+".isida-9-"+str(l)+"-"+str(u)+".svm",zero_based=False)
                    except:
                        dump_svmlight_file(desc, data_table[p], 
                            outdir+'/'+p+".isida-9-"+str(l)+"-"+str(u)+".svm",zero_based=False)
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
                    try:
                        dump_svmlight_file(desc, data_table[p], 
                            outdir+'/'+args.property_names[i]+".isida-6-"+str(l)+"-"+str(u)+".svm",zero_based=False)
                    except:
                        dump_svmlight_file(desc, data_table[p], 
                            outdir+'/'+p+".isida-6-"+str(l)+"-"+str(u)+".svm",zero_based=False)

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
                    try:
                        dump_svmlight_file(desc, data_table[p], 
                            outdir+'/'+args.property_names[i]+".circus-"+str(l)+"-"+str(u)+".svm",zero_based=False)
                    except:
                        dump_svmlight_file(desc, data_table[p], 
                            outdir+'/'+p+".circus-"+str(l)+"-"+str(u)+".svm",zero_based=False)