# -*- coding: utf-8 -*-
#
#  Copyright 2021 Pavel Sidorov <pavel.o.sidorov@gmail.com>
#  This file is part of ChemInfoTools repository.
#
#  ChemInfoTools is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.

import pandas as pd
from pandas import DataFrame
import numpy as np
from numpy import array
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectorMixin 
from chython import smiles, CGRContainer, MoleculeContainer, from_rdkit_molecule, to_rdkit_molecule
from typing import Optional, List, Dict, Tuple
from rdkit import Chem
from rdkit.Chem import MACCSkeys, AllChem
from rdkit.DataStructs.cDataStructs import ExplicitBitVect
from rdkit.Chem import rdMolDescriptors
from rdkit.Avalon import pyAvalonTools

class ChythonCircus(BaseEstimator, TransformerMixin):
    """
    ChythonCircus class is a scikit-learn compatible transformer that calculates the fragment features 
    from molecules and Condensed Graphs of Reaction (CGR). The features are augmented substructures - 
    atom-centered fragments that take into account atom and its environment. Implementation-wise,
    this takes all atoms in the molecule/CGR, and builds topological neighborhood spheres around them.
    All atoms and bonds that are in a sphere of certain radius (1 bond, 2 bonds, etc) are taken into 
    the substructure. All such substructures are detected and stored as distinct features. The 
    substructures will keep any rings found within them. The value of the feature is the number of
    occurrence of such substructure in the given molecule.

    The parameters of the augmentor are the lower and the upper limits of the radius. By default,
    both are set to 1, which means only the count of atoms.
    Additionally, only_dynamic flag indicates of only fragments that contain a dynamic bond or atom 
    will be considered (only works in case of CGRs).
    fmt parameter defines the format in which the molecules are given to the ChythonCircus. 
    "mol" if they are in CGRtools MoleculeContainer or CGRContainer, "smiles" if they are in SMILES.
    """

    def __init__(self, lower:int=1, upper:int=1, only_dynamic:bool=False, fmt:str="mol"): 
        self.feature_names = []
        self.lower = lower 
        self.upper = upper
        self.only_dynamic = only_dynamic
        self.fmt = fmt
    
    def fit(self, X:DataFrame, y:Optional[List]=None):
        """Fits the augmentor - finds all possible substructures in the given array of molecules/CGRs.

        Parameters
        ----------
        X : array-like, [MoleculeContainers, CGRContainers]
            the  array/list/... of molecules/CGRs to train the augmentor. Collects all possible substructures.

        y : None
            required by default by scikit-learn standards, but doesn't change the function at all.

        Returns
        -------
        None
        """
        output = []
        for i, mol in enumerate(X):
            if self.fmt == "smiles":
                mol = smiles(mol)
            output.append(mol.morgan_smiles_hash(self.lower, self.upper))
        self.feature_names = pd.DataFrame(output).columns
        return self
                        # if dynamic_only is on, skip all non-dynamic fragments
                        #if self.only_dynamic and ">" not in sub:
                        #    continue
                        #self.feature_names.append(sub)
        #return self

    def transform(self, X:DataFrame, y:Optional[List]=None) -> DataFrame:
        """Transforms the given array of molecules/CGRs to a data frame with features and their values.

        Parameters
        ----------
        X : array-like, [MoleculeContainers, CGRContainers]
            the  array/list/... of molecules/CGRs to transform to feature table using trained feature list.

        y : None
            required by default by scikit-learn standards, but doesn't change the function at all.

        Returns
        -------
        DataFrame containing the fragments and their counts.
        """
        df = pd.DataFrame(columns=self.feature_names, dtype=int)

        output = []
        for m in X:
            all_bits = sum([list(v.values()) for v in m._morgan_hash_dict(self.lower, self.upper)], [])
            smiles_bits = m.morgan_smiles_hash(self.lower, self.upper)
            tmp = {}
            for k,v in m.morgan_smiles_hash(1,5).items():
                tmp[k] = 0
                for vv in v:
                    tmp[k] += all_bits.count(vv)
            output.append(tmp)
        output = pd.DataFrame(output)
        
        output2 = output[output.columns.intersection(df.columns)]
        df = pd.concat([df, output2])
        return df
    
    def get_feature_names(self) -> List[str]:
        """Returns the list of features as strings.

        Returns
        -------
        List[str]
        """
        return list(self.feature_names)

class ChythonLinear(BaseEstimator, TransformerMixin):
    def __init__(self, lower:int=0, upper:int=0, only_dynamic:bool=False, fmt:str="mol"): 
        self.feature_names = []
        self.lower = lower 
        self.upper = upper
        self.only_dynamic = only_dynamic
        self.fmt = fmt

    def fit(self, X:DataFrame, y:Optional[List]=None):
        """Fits the linear fragmentor - finds all possible substructures in the given array of molecules/CGRs.

        Parameters
        ----------
        X : array-like, [MoleculeContainers, CGRContainers]
            the  array/list/... of molecules/CGRs to train the augmentor. Collects all possible substructures.

        y : None
            required by default by scikit-learn standards, but doesn't change the function at all.

        Returns
        -------
        None
        """
        output = []
        for i, mol in enumerate(X):
            if self.fmt == "smiles":
                mol = smiles(mol)
            output.append(mol.linear_smiles_hash(self.lower, self.upper, number_bit_pairs=0))
        self.feature_names = pd.DataFrame(output).columns
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(columns=self.feature_names, dtype=int)

        output = []
        for m in X:
            output.append(m.linear_smiles_hash(self.lower, self.upper, number_bit_pairs=0))
        output = pd.DataFrame(output)
        output = output.map(lambda x: len(x) if isinstance(x, list) else 0)
        output = output.fillna(0)
        
        output2 = output[output.columns.intersection(df.columns)]
        df = pd.concat([df, output2])
        return df

    def get_feature_names(self) -> List[str]:
        """Returns the list of features as strings.

        Returns
        -------
        List[str]
        """
        return list(self.feature_names)

class ComplexFragmentor(BaseEstimator, TransformerMixin):
    """
    ComplexFragmentor class is a scikit-learn compatible transformer that concatenates the features 
    according to specified associations. The most important argument is the "associator" - a dictionary
    that establishes the correspondence between a column in a data frame X and the transformer 
    that is trained on it.

    For example, say you have a data frame with molecules/CGRs in one column ("molecules"), and 
    solvents in another ("solvent"). You want to generate a feture table that includes both structural 
    and solvent descriptors. You would define a ComplexFragmentor class with associator as a dictionary,
    where keys are column names, and value are the corresponding feature generators. In this case, e.g.,

        associator = {"molecules": Augmentor(lower=a, upper=b),
                      "solvent":SolventVectorizer()}  # see CIMTools library for solvent features

    ComplexFragmentor assumes that one of the types of features will be structural, thus, 
    "structure_column" parameter defines the column of the data frame where structures are found.
    """
    def __init__(self, associator:Dict[str,object], structure_columns:List[str]=[]):
        self.associator = associator
        self.structure_columns = structure_columns
        #self.fragmentor = self.associator[self.structure_column]
        self.feature_names = []
        
    def get_feature_names(self) -> List[str]:
        """Returns the list of all features as strings.

        Returns
        -------
        List[str]
        """
        return self.feature_names
    
    def get_structural_feature_names(self) -> List[str]:
        """Returns the list of only structural features associated to the structure_column as strings.

        Returns
        -------
        List[str]
        """
        return self.fragmentor.get_feature_names()
    
    def fit(self, x:DataFrame, y:Optional[List]=None):
        """Fits the ComplexFragmentor - fits all feature generators separately, then concatenates them.

        Parameters
        ----------
        X : array-like, [MoleculeContainers, CGRContainers]
            the data frame to train all feature generators. Must contain all columns indicated in 
            the associator.

        y : None
            required by default by scikit-learn standards, but doesn't change the function at all.

        Returns
        -------
        None
        """
        for k, v in self.associator.items():
            v.fit(x[k])
            self.feature_names += v.get_feature_names()
        return self
    
    def transform(self, x:DataFrame) -> DataFrame:
        """Transforms the given data frame to a data frame of features with their values.
        Applies each feature generator separately, then concatenates them.

        Parameters
        ----------
        X : DataFrame
            the data frame to transform to feature table using trained feature list. Must contain 
            columns indicated in the associator.

        y : None
            required by default by scikit-learn standards, but doesn't change the function at all.

        Returns
        -------
        DataFrame containing the fragments and their counts and other descriptors.
        """
        concat = []
        for k, v in self.associator.items():
            if len(x.shape) == 1:
                concat.append(v.transform([x[k]]))
            else:
                concat.append(v.transform(x[k]))
        return pd.concat(concat, axis=1, sort=False)

class PassThrough(BaseEstimator, TransformerMixin):
    """
    PassThrough is a sklearn-compatible transformer that passes a column from a Dataframe into the 
    feature Dataframe without any changes. It is functionally identical to sklearn.compose 
    ColumnTransformer's passthrough function. Needed to be compatible with ComplexFragmentor.
    
    """
    def __init__(self, column_name:str):
        self.column_name = column_name
        self.feature_names = [self.column_name]
        
    def get_feature_names(self):
        """Returns the list of all features as strings.

        Returns
        -------
        List[str]
        """
        return self.feature_names
    
    def fit(self, x:DataFrame, y=None):
        """Fits the ComplexFragmentor - fits all feature generators separately, then concatenates them.

        Parameters
        ----------
        X : array-like, DataFrame
            must contain the column taken as self.column_name

        y : None
            required by default by scikit-learn standards, but doesn't change the function at all.

        Returns
        -------
        None
        """
        return self
    
    def transform(self, x:DataFrame):
        """Transforms the given data frame to a data frame of features with their values.
        Applies each feature generator separately, then concatenates them.

        Parameters
        ----------
        X : DataFrame
            the data frame to transform to feature table using trained feature list. Must contain 
            columns indicated in the associator.

        y : None
            required by default by scikit-learn standards, but doesn't change the function at all.

        Returns
        -------
        DataFrame contaning the column.
        """
        return pd.Series(x, name=self.column_name)

class Fingerprinter(BaseEstimator, TransformerMixin):
    def __init__(self, fp_type, nBits=1024, size=None, params={}):
        self.fp_type = fp_type
        self.nBits = nBits
        self.size = size
        self.params = params
        self.info = dict([(i, []) for i in range(self.nBits)])
        self.feature_names = dict([(i, []) for i in range(self.nBits)])
        self.feature_names_chython = dict([(i, []) for i in range(self.nBits)])
        
    def fit(self, X, y=None):
        if self.fp_type=='morgan':
            for x in X:
                temp = {}
                m = Chem.MolFromSmiles(str(x))
                AllChem.GetMorganFingerprintAsBitVect(m, 
                                                      nBits=self.nBits, 
                                                      radius=self.size, 
                                                      bitInfo=temp, 
                                                      **self.params)
                self.info.update(temp)
                for k, v in temp.items():
                    for i in v:
                        if i[1]>0:
                            env = Chem.FindAtomEnvironmentOfRadiusN(m,i[1],i[0])
                            amap={}
                            submol=Chem.PathToSubmol(m,env,atomMap=amap)
                            self.feature_names[k].append(Chem.MolToSmiles(submol,canonical=True))
                        else:
                            self.feature_names[k].append(m.GetAtomWithIdx(i[0]).GetSymbol())
                        #self.feature_names_chython[k].append(str(x.augmented_substructure([i[0]+1], deep=i[1])))
            for k, v in self.feature_names.items():
                vt = [item for item in v if item != '']
                self.feature_names[k] = set(vt)
            #for k, v in self.feature_names_chython.items():
            #    vt = [item for item in v if item != '']
            #    self.feature_names_chython[k] = set(vt)
        elif self.fp_type == 'avalon':
            pass
        elif self.fp_type == 'layered':
            pass
        elif self.fp_type == 'ap':
            pass
        elif self.fp_type == 'torsion':
            pass
        elif self.fp_type=='rdkfp':
            for x in X:
                temp = {}
                m = Chem.MolFromSmiles(str(x))
                Chem.RDKFingerprint(m, fpSize=self.nBits, useHs=False,
                                    maxPath=self.size, bitInfo=temp, **self.params)
                self.info.update(temp)
                for k, v in temp.items():
                    for i in v:
                        self.feature_names[k].append(Chem.MolFragmentToSmiles(m,
                                                    atomsToUse=set(sum([[m.GetBondWithIdx(b).GetBeginAtomIdx(),
                                                    m.GetBondWithIdx(b).GetEndAtomIdx()] for b in i], [])),
                                                     bondsToUse=i))
            for k, v in self.feature_names.items():
                vt = [item for item in v if item != '']
                self.feature_names[k] = set(vt)
            #self.feature_names_chython = self.feature_names
        
        return self
        
    def get_features(self, x):
        features = dict([(i, []) for i in range(self.nBits)])
        if self.fp_type=='morgan':
            m = Chem.MolFromSmiles(str(x))
            temp = {} 
            AllChem.GetMorganFingerprintAsBitVect(m, 
                                                nBits=self.nBits, 
                                                radius=self.size, 
                                                bitInfo=temp, 
                                                **self.params)
            for k, v in temp.items():
                for i in v:
                    features[k].append(str(m.augmented_substructure([i[0]+1], deep=i[1])))
            for k, v in features.items():
                vt = [item for item in v if item != '']
                features[k] = set(vt)
        elif self.fp_type == 'rdkfp':
            temp = {}
            m = Chem.MolFromSmiles(str(x))
            Chem.RDKFingerprint(m, fpSize=self.nBits, useHs=False,
                                maxPath=self.size, bitInfo=temp, **self.params)
            for k, v in temp.items():
                for i in v:
                    features[k].append(Chem.MolFragmentToSmiles(m,
                                            atomsToUse=set(sum([[m.GetBondWithIdx(b).GetBeginAtomIdx(),
                                            x.GetBondWithIdx(b).GetEndAtomIdx()] for b in i], [])),
                                            bondsToUse=i))
            for k, v in self.feature_names.items():
                vt = [item for item in v if item != '']
                features[k] = set(vt)
        elif self.fp_type == 'avalon':
            pass
        elif self.fp_type == 'layered':
            pass
        elif self.fp_type == 'ap':
            pass
        elif self.fp_type == 'torsion':
            pass
        return features

    def get_feature_names(self):
        return([str(i) for i in range(self.nBits)])
                                       
    def transform(self, X, y=None):
        res = []
        for x in X:
            m = Chem.MolFromSmiles(str(x))
            if self.fp_type=='morgan':
                res.append(AllChem.GetMorganFingerprintAsBitVect(m, 
                                                                 nBits=self.nBits, 
                                                                 radius=self.size, 
                                                                 **self.params))
            if self.fp_type=='avalon':
                res.append(pyAvalonTools.GetAvalonFP(m, nBits=self.nBits))
            if self.fp_type=='rdkfp':
                res.append(Chem.RDKFingerprint(m, fpSize=self.nBits, useHs=False,
                                maxPath=self.size, **self.params))
            if self.fp_type=='layered':
                res.append(Chem.LayeredFingerprint(m, fpSize=self.nBits, 
                                maxPath=self.size, **self.params))
            if self.fp_type=='ap':
                res.append(rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(m, nBits=self.nBits, 
                                **self.params))
            if self.fp_type=='torsion':
                res.append(rdMolDescriptors.GetHashedTopologicalTorsionFingerprintAsBitVect(m, nBits=self.nBits, 
                                **self.params))
            
        return pd.DataFrame(np.array(res), columns=[str(i) for i in range(self.nBits)])
    

class Pruner(BaseEstimator, SelectorMixin, TransformerMixin):
    """
    Pruner is a feature selecter and scaler. It is meant to work ONLY with the genetic algorithm 
    optimization of SVM models described elsewhere. I does not do pruning/scaling by itself!!!
    Requires a .pri file generated by the above-mentioned GA implemetation.
    """
    def __init__(self, prifile:str, scaling:bool=True):
        self.prifile = prifile
        self.scaling = scaling
        
        self.indices = np.loadtxt(prifile)[:,0].astype(int)
        self.max_len = np.max(self.indices)
        self.scale =  np.loadtxt(prifile)[:,[2,3]]
        
    def fit(self, X:DataFrame, y:Optional[List]=None):
        self.max_len = X.shape[1]
        return self
    
    def transform(self, X:DataFrame) -> DataFrame:
        features = np.array(X.columns)[self.indices-1]
        X = np.array(X)[:, self.indices-1]
        if self.scaling:
            X = (X - self.scale[:,0])/(self.scale[:,1]-self.scale[:,0])
        return pd.DataFrame(X, columns=features)
    
    def _get_support_mask(self) -> array:
        mask = np.zeros(self.max_len)
        mask[self.indices] += 1
        return mask.astype(bool)