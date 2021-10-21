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
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from CGRtools import smiles, CGRContainer, MoleculeContainer

class Augmentor(BaseEstimator, TransformerMixin):
    """
    Augmentor class is a scikit-learn compatible transformer that calculates the fragment features 
    from molecules and Condensed Graphs of Reaction (CGR). The features are augmented substructures - 
    atom-centered fragments that take into account atom and its environment. Implementation-wise,
    this takes all atoms in the molecule/CGR, and builds topological neighborhood spheres around them.
    All atoms and bonds that are in a sphere of certain radius (1 bond, 2 bonds, etc) are taken into 
    the substructure. All such substructures are detected and stored as distinct features. The 
    substructures will keep any rings found within them. The value of the feature is the number of
    occurrence of such substructure in the given molecule.

    The parameters of the augmentor are the lower and the upper limits of the radius. By default,
    both are set to 0, which means only the count of atoms.
    Additionally, only_dynamic flag indicates of only fragments that contain a dynamic bond or atom 
    will be considered (only works in case of CGRs).
    """

    def __init__(self, lower:int=0, upper:int=0, only_dynamic:bool=False):
        self.feature_names = []
        self.lower = lower 
        self.upper = upper
        self.only_dynamic = only_dynamic
    
    def fit(self, X, y=None):
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
        for i, mol in enumerate(X):
            for length in range(self.lower, self.upper):
                for atom in mol.atoms():
                    # deep is the radius of the neighborhood sphere in bonds
                    sub = str(mol.augmented_substructure([atom[0]], deep=length))
                    if sub not in self.feature_names:
                        # if dynamic_only is on, skip all non-dynamic fragments
                        if self.only_dynamic and ">" not in sub:
                            continue
                        self.feature_names.append(sub)
        return self
        
    def transform(self, X, y=None):
        """Transforms the given array of molecules/CGRs to a data frame with features and their values.

        Parameters
        ----------
        X : array-like, [MoleculeContainers, CGRContainers]
            the  array/list/... of molecules/CGRs to transform to feature table using trained feature list.

        y : None
            required by default by scikit-learn standards, but doesn't change the function at all.

        Returns
        -------
        None
        """
        table = pd.DataFrame(columns=self.feature_names)
        for i, mol in enumerate(X):
            table.loc[len(table)] = 0
            for sub in self.feature_names:
                # if CGRs are used, the transformation of the substructure to the CGRcontainer is needed
                if type(mol) == CGRContainer:
                    mapping = list(CGRContainer().compose(smiles(sub)).get_mapping(mol, optimize=False))
                else:
                    mapping = list(smiles(sub).get_mapping(mol, optimize=False))
                # mapping is the list of all possible substructure mappings into the given molecule/CGR
                table.loc[i,sub] = len(mapping)
        return table
    
    def get_feature_names(self):
        """Returns the list of features as strings.

        Returns
        -------
        List[str]
        """
        return self.feature_names

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
    def __init__(self, associator, structure_columns=[]):
        self.associator = associator
        self.structure_columns = structure_columns
        #self.fragmentor = self.associator[self.structure_column]
        self.feature_names = []
        
    def get_feature_names(self):
        """Returns the list of all features as strings.

        Returns
        -------
        List[str]
        """
        return self.feature_names
    
    def get_structural_feature_names(self):
        """Returns the list of only structural features associated to the structure_column as strings.

        Returns
        -------
        List[str]
        """
        return self.fragmentor.get_feature_names()
    
    def fit(self, x, y=None):
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
    
    def transform(self, x):
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
        None
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
    
    def fit(self, x, y=None):
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
    
    def transform(self, x):
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
        None
        """
        return pd.Series(x, name=self.column_name)