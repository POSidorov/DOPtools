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
from CIMtools.preprocessing.fragmentor import Fragmentor
from CGRtools import smiles as cgrtools_smiles

class ChythonIsida(BaseEstimator, TransformerMixin):
    def __init__(self, ftype=3, lower=2, upper=10, cgr_dynbonds=0, doallways=False,
                 useformalcharge=False, header=None, workpath='.', version='2017',
                 verbose=False, remove_rare_ratio=0, return_domain=False):
        self.fragment_type = ftype
        self.min_length = lower
        self.max_length = upper
        self.cgr_dynbonds = cgr_dynbonds
        self.doallways = doallways
        self.useformalcharge = useformalcharge
        self.version = version
        self.verbose = verbose
        self.header = header
        self.remove_rare_ratio = remove_rare_ratio
        self.return_domain = return_domain
        self.fragmentor = Fragmentor(fragment_type=self.fragment_type, min_length=self.min_length, 
                                     max_length=self.max_length, cgr_dynbonds=self.cgr_dynbonds, 
                                     doallways=self.doallways, useformalcharge=self.useformalcharge, 
                                     version=self.version, verbose=self.verbose, header=self.header, 
                                     remove_rare_ratio=self.remove_rare_ratio, return_domain=self.return_domain)
        
    def fit(self, X, y=None):
        x = [cgrtools_smiles(str(s)) for s in X]
        self.fragmentor.fit(x, y)
        return self
        
    def transform(self, X):
        x = [cgrtools_smiles(str(s)) for s in X]
        return self.fragmentor.transform(x)
        
    def get_feature_names(self):
        return self.fragmentor.get_feature_names()