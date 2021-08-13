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

from CGRtools import RDFRead, ReactionContainer, SDFRead, SMILESRead, smiles
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline

class ColorAtom:
    def __init__(self, fragmentor=None, model=None):
        self.model = model
        self.pipeline = None
        self.fragmentor = fragmentor
        self.descriptors = []
        self.isida_like = False
    
    def set_pipeline(self, pipeline, fragmentor_pos_in_pipeline=0):
        self.pipeline = pipeline
        if self.fragmentor is None:
            self.fragmentor = self.pipeline[fragmentor_pos_in_pipeline]
            self.descriptors = np.array(self.fragmentor.get_feature_names())
        if self.descriptors[0].split(",")[-1].startswith("x"):
            self.isida_like = True
        
    def calculate_atom_contributions(self, mol):
        atom_weights = {i[0]:0 for i in mol.atoms()}
        mol_descriptors = self.pipeline["fragmentor"].transform([mol])
        true_prediction = self.pipeline.predict([mol])[0]
        for i, d in enumerate(self.initial_descriptors):
            new_line = np.array(mol_descriptors)[0].copy()
            if new_line[i]>0:
                new_line[i] -= 1 
            new_prediction = self.pipeline[1:].predict([new_line])[0]
            w = true_prediction - new_prediction

            if w != 0:
                if self.isida_like:
                    participating_atoms = self._full_mapping_from_descriptor(mol, d)
                else:
                    import itertools
                    participating_atoms = [list(i.values()) for i in smiles(d).get_mapping(mol, optimize=False)]
                    participating_atoms = set(list(itertools.chain.from_iterable(participating_atoms)))
                for a in participating_atoms:
                    atom_weights[a] -= w
        return atom_weights
        
    def _full_mapping_from_descriptor(self, mol, isida_fragment):
        subfragments = isida_fragment.split(",")
        central_atom = subfragments[-1][-1]
        subfragments = [i[1:-1] for i in subfragments[:-1]]

        full_mapping = []

        for atom in mol.atoms():
            if atom[1].atomic_symbol == central_atom:
                struct = mol.augmented_substructure([atom[0]], deep=len(self._only_atoms(subfragments[0]))-1)
                for frag in set(subfragments):
                    needed_count = subfragments.count(frag)
                    real_frags = list(smiles(self._aromatize(frag)).get_mapping(struct, optimize=False))
                    real_frags = [i for i in real_frags if i[1] == atom[0]]
                    real_count = len(real_frags)
                    if needed_count < real_count:
                        for f in real_frags:
                            last_atom = f[len(only_atoms(subfragments[0]))]
                            smaller_struct = mol.augmented_substructure([atom[0]], deep=len(self._only_atoms(subfragments[0]))-2)
                            if last_atom in [j[0] for j in smaller_struct.atoms()]:
                                real_count -= 1
                    if needed_count == real_count:
                        for f in real_frags:
                            full_mapping += f.values()
        return set(full_mapping)
    
    def _aromatize(self, text):
        res = list(text)
        for i, symbol in enumerate(res):
            if symbol=="*":
                res[i-1]=res[i-1].lower()

                res[i+1]=res[i+1].lower()

        for i, symbol in enumerate(res):
            if symbol=="*":
                res[i] = ""

        return("".join(res))

    def _only_atoms(self, text):
        res = list(text)
        for i, symbol in enumerate(res):
            if symbol=="*" or symbol=="=" or symbol=="-" or symbol=="#":
                res[i]=""
        return "".join(res)