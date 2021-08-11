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
    def __init__(self, associator, structure_column="molecules"):
        self.associator = associator
        self.structure_column = structure_column
        self.fragmentor = self.associator[self.structure_column]
        self.feature_names = []
        
    def get_feature_names(self):
        """Returns the list of all features as strings.

        Returns
        -------
        List[str]
        """
        return self.feature_names
    
    def get_structural_feature_names(self):
        """Returns the list of only structural features as strings.

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