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

from chython import ReactionContainer, MoleculeContainer, CGRContainer, smiles
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from chython.algorithms import depict as DEPICT
from IPython.display import HTML
from matplotlib.cm import RdYlGn, PiYG
from matplotlib.colors import rgb2hex
import itertools

class ColorAtom:
    """
    ColorAtom class implements the approach of calculating atomic contributions to the prediction
    by a model built using fragment descriptors. The approach is based on the approximation that the
    weight of a fragment can be calculated as a partial derivative of the model's prediction, and the 
    atoms in the fragment contribute equally to this weight. The approach is developed and reported in 

    G. Marcou, D. Horvath, V. Solov’ev, A. Arrault, P. Vayer and A. Varnek
    Interpretability of SAR/QSAR models of any complexity by atomic contributions
    Mol. Inf., 2012, 31(9), 639-642, 2012

    Current implementation is designed for regression tasks, for models built with Scikit-learn library and
    using CircuS fragments implemented in this library.
    """
    def __init__(self, fragmentor=None, model=None, is_complex:bool=False, isida_like:bool=False):
        DEPICT.depict_settings(monochrome=True, aam=False)
        self.model = model
        self.pipeline = None
        self.fragmentor = fragmentor
        self.descriptors = []
        self.isida_like = isida_like
        self.complex = is_complex
    
    def set_pipeline(self, pipeline:Pipeline, fragmentor_pos_in_pipeline:int=0):
        """Sets the fragmentor and model of the ColorAtom class via sklearn Pipeline. The fragmentor of
        the ColorAtom object is set as the nth position in the pipeline, the model as the rest.

        Parameters
        ----------
        pipeline : sklearn Pipeline class
            the pipeline containing fragmentor, preprocessing and the model

        fragmentor_pos_in_pipeline : int
            the position of the fragmentor in the pipeline, 0 by default

        Returns
        -------
        None
        """
        self.pipeline = pipeline
        if self.fragmentor is None:
            self.fragmentor = self.pipeline[fragmentor_pos_in_pipeline]
            self.descriptors = np.array(self.fragmentor.get_feature_names())
            self.model = Pipeline([p for i, p in enumerate(pipeline.steps) if i != fragmentor_pos_in_pipeline])
        
    def calculate_atom_contributions(self, mol, algo="derivative"):
        """Calculates the atom contribution with the partial derivative approach for the given molecule.
        If the fragmentor is an object of ComplexFragmentor class, a dataframe with the columns required 
        by the ComplexFragmentor is accepted. In the latter case, atom contributions will be calculated 
        for each structural column in the ComplexFragmentor.

        Parameters
        ----------
        mol : [MoleculeContainer,CGRContainer,DataFrame]
            the molecule for which the atom contributions will be calculated

        Returns
        -------
        atom_weights: Dict[MoleculeContainer: Dict[int:float]]
            dictionary in form {Molecule: {atom1:weight1, atom2:weight2, ...}}
        """
        if self.complex:
            atom_weights = {}
            total_descs = 0
            descriptor_vector = []
            for a, b in self.fragmentor.associator.items():
                descriptor_vector.append(b.transform([mol[a]]))
            descriptor_vector = pd.concat(descriptor_vector, axis=1)
            if algo=="derivative":
                true_prediction = self.model.predict(descriptor_vector)[0]
                for col, fragmentor in self.fragmentor.associator.items():
                    if col in self.fragmentor.structure_columns:
                        m = mol[col]
                        atom_weights[m] = {i[0]:0 for i in m.atoms()}

                        for i, d in enumerate(fragmentor.get_feature_names()):
                            new_line = descriptor_vector.copy()
                            if new_line.iloc[0,total_descs+i]>0:
                                new_line.at[0,d] -= 1 
                            new_prediction = self.model.predict(new_line)[0]
                            w = true_prediction - new_prediction

                            if w != 0:
                                if self.isida_like:
                                    if type(m) == CGRContainer:
                                        d = self._isida2cgrtools(d)
                                    participating_atoms = self._full_mapping_from_descriptor(m, d)
                                else:
                                    if "*" in d:
                                        d = self._aromatize(d)
                                    if type(m) == CGRContainer:
                                        participating_atoms = [list(i.values()) for i in CGRContainer().compose(smiles(d)).get_mapping(m, optimize=False)]
                                    else:
                                        participating_atoms = [list(i.values()) for i in smiles(d).get_mapping(m,)]
                                    participating_atoms = set(list(itertools.chain.from_iterable(participating_atoms)))
                                for a in participating_atoms:
                                    atom_weights[m][a] += w
                    total_descs += len(fragmentor.get_feature_names())
            #elif algo=="shap":
                
                    
        else:
            atom_weights = {mol:{i[0]:0 for i in mol.atoms()}}
            descriptor_vector = self.fragmentor.transform([mol])
            true_prediction = self.model.predict(descriptor_vector)[0]
            for i, d in enumerate(self.descriptors):
                new_line = descriptor_vector.copy()
                if new_line.iloc[0,i]>0:
                    new_line.at[0,d] -= 1 
                new_prediction = self.model.predict(new_line)[0]
                w = true_prediction - new_prediction

                if w != 0:
                    if self.isida_like:
                        d = self._isida2cgrtools(d)
                        participating_atoms = self._full_mapping_from_descriptor(mol, d)
                    else:
                        if "*" in d:
                            d = self._aromatize(d)
                        participating_atoms = [list(i.values()) for i in smiles(d).get_mapping(mol)]
                        participating_atoms = set(list(itertools.chain.from_iterable(participating_atoms)))
                    for a in participating_atoms:
                        atom_weights[mol][a] += w
        return atom_weights
    
    def output_html(self, mol, contributions=None):
        """For the given molecule (DataFrame if complex), generates the SVG image where the contributions
        of atoms are depicted with colors (purple for negative contributions, green for positive, by default).
        The depicition is based on the CGRTools Depict class and method.
        The method returns an HTML object which contains as many pictures, as there are molecules to depict.

        Parameters
        ----------
        mol : [MoleculeContainer,CGRContainer,DataFrame]
            the molecule for which the image of atom contributions will be generated

        contributions: Dict [optional]
            if given, the contribution of the molecule will not be recalculated

        Returns
        -------
        html: HTML object
            an HTML object containing SVG image for each structure with contributions colored
        """
        if contributions is None:
            contributions = self.calculate_atom_contributions(mol)
        if self.complex:
            max_value = np.max(np.abs(np.concatenate([list(i.values()) for i in list(contributions.values())])))
        else:
            max_value = np.max(np.abs(list(contributions[mol].values())))
            
        svgs = []
        for m in contributions.keys():
            ext_svg = m.depict()[:-6]
            ext_svg = '<svg style="background-color:white" '+ext_svg[4:]
            for k, c in contributions[m].items():
                x, y = m.atom(k).xy[0], -m.atom(k).xy[1]
                if len(m.atom(k).atomic_symbol) >1:
                    x -= 0.1
                color = rgb2hex(PiYG((c+max_value)/2./max_value))
                ext_svg += '<circle cx="{}" cy="{}" r="0.33" stroke="{}" stroke-width="0.1" fill="none" />'.format(x, y, color)
            ext_svg += "</svg>"
            svgs.append(ext_svg)
        no_wrap_div = '<div style="white-space: nowrap">'+'{}'*len(svgs)+'</div>'
        return HTML(no_wrap_div.format(*svgs))
        
    def _full_mapping_from_descriptor(self, mol, isida_fragment):
        subfragments = isida_fragment.split(",")
        central_atom = subfragments[-1][1:]
        subfragments = [i[1:-1] for i in subfragments[:-1]]

        full_mapping = []

        for atom in mol.atoms():
            if atom[1].atomic_symbol == central_atom:
                struct = mol.augmented_substructure([atom[0]], deep=len(self._only_atoms(subfragments[0]))-1)
                for frag in set(subfragments):
                    needed_count = subfragments.count(frag)
                    real_frags = list(smiles(self._aromatize(frag)).get_mapping(struct))
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

    def _isida2cgrtools(self, text):
        res = list(text)
        for i, symbol in enumerate(res):
            if symbol=="+" :
                res[i]="#"
        text = "".join(res)
        text = text.replace("2>1", "[=>-]").replace("2>3", "[=>#]").replace("2>0", "[=>.]")
        text = text.replace("1>2", "[->=]").replace("1>3", "[->#]").replace("1>0", "[->.]")
        text = text.replace("3>2", "[#>=]").replace("3>1", "[#>-]").replace("3>0", "[#>.]")
        text = text.replace("0>2", "[.>=]").replace("0>3", "[.>#]").replace("0>1", "[.>-]")
        return text