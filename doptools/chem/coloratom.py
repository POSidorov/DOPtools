# -*- coding: utf-8 -*-
#
#  Copyright 2022-2025 Pavel Sidorov <pavel.o.sidorov@gmail.com> This
#  file is part of DOPTools repository.
#
#  DOPtools is free software; you can redistribute it and/or modify
#  it under the terms of the GNU Lesser General Public License as published by
#  the Free Software Foundation; either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU Lesser General Public License for more details.
#
#  You should have received a copy of the GNU Lesser General Public License
#  along with this program; if not, see <https://www.gnu.org/licenses/>.


from chython import ReactionContainer, MoleculeContainer, CGRContainer, smiles
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn import base
from chython.algorithms import depict as DEPICT
from IPython.display import HTML
from matplotlib.cm import RdYlGn, PiYG, Blues
from matplotlib.colors import rgb2hex
import itertools
from io import StringIO
from doptools.chem.chem_features import ComplexFragmentor
from typing import List
from pandas import DataFrame, Series
import matplotlib.pyplot as plt
import matplotlib as mpl

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
    def __init__(self, fragmentor=None, model=None, is_complex:bool=False, structure_cols:List=None, 
                 colormap=None, reaction=None):
        DEPICT.depict_settings(monochrome=True, aam=False)
        self.model = model
        self.pipeline = None
        self.model_type = "R"
        self.fragmentor = fragmentor
        self.descriptors = []
        #self.isida_like = isida_like
        if isinstance(self.fragmentor, ComplexFragmentor):
            self.complex = True
        else:
            self.complex = is_complex
        self.reaction = reaction
        self.structure_cols = structure_cols
        self.colormap = colormap
    
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
        if isinstance(self.fragmentor, ComplexFragmentor):
            self.complex = True
            if self.structure_cols is None:
                self.structure_cols = list(set([d.split("::")[0] for d in self.descriptors]))
        self.model = Pipeline([p for i, p in enumerate(pipeline.steps) if i != fragmentor_pos_in_pipeline])
        if issubclass(self.pipeline[-1].__class__, base.ClassifierMixin):
            self.model_type = "C"
            if self.colormap is None:
                self.colormap = Blues
        else:
            if self.colormap is None:
                self.colormap = PiYG
        
    def calculate_atom_contributions(self, mol):
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
            for m in self.structure_cols:
                if isinstance(mol[m], ReactionContainer):
                    atom_weights[mol[m]] = {i[0]:0 for react in mol[m].molecules() for i in react.atoms()}
                else:
                    atom_weights[mol[m]] = {i[0]:0 for i in mol[m].atoms()}
        else:
            if isinstance(mol, ReactionContainer):
                atom_weights[mol] = {i[0]:0 for react in mol.molecules() for i in react.atoms()}
            else:
                atom_weights = {mol:{i[0]:0 for i in mol.atoms()}}
        if not isinstance(mol, Series):
            descriptor_vector = self.fragmentor.transform([mol])
        else:
            descriptor_vector = self.fragmentor.transform(mol)
        if self.model_type=="R":
            true_prediction = self.model.predict(descriptor_vector)[0]
            for i, d in enumerate(self.descriptors):
                new_line = descriptor_vector.copy()
                if new_line.iloc[0,i]>0:
                    new_line.at[0,d] -= 1 
                new_prediction = self.model.predict(new_line)[0]
                w = true_prediction - new_prediction

                if w != 0:
                    #if self.isida_like:
                    #    d = self._isida2cgrtools(d)
                    #    participating_atoms = self._full_mapping_from_descriptor(mol, d)
                    #else:
                    if "*" in d:
                        d = self._aromatize(d)
                    if self.complex:
                        mol_name, frag_smiles = d.split("::")
                        if mol_name not in self.structure_cols:
                            continue
                        if isinstance(mol[mol_name], ReactionContainer):
                            react_cgr = mol[mol_name].compose()
                            frag_cgr = self._frag2cgr(frag_smiles)
                            participating_atoms = [list(i.values()) for i in frag_cgr.get_mapping(react_cgr)]
                        else:
                            participating_atoms = [list(i.values()) for i in smiles(frag_smiles).get_mapping(mol[mol_name])]
                        participating_atoms = set(list(itertools.chain.from_iterable(participating_atoms)))
                        for a in participating_atoms:
                            atom_weights[mol[mol_name]][a] += w
                    else:
                        if isinstance(mol, ReactionContainer):
                            react_cgr = mol.compose()
                            frag_cgr = self._frag2cgr(d)
                            participating_atoms = [list(i.values()) for  i in frag_cgr.get_mapping(react_cgr)]
                        else:
                            participating_atoms = [list(i.values()) for i in smiles(d).get_mapping(mol)]
                        participating_atoms = set(list(itertools.chain.from_iterable(participating_atoms)))
                        for a in participating_atoms:
                            atom_weights[mol][a] += w
        elif self.model_type=="C":
            true_prediction = self.model.predict(descriptor_vector)[0]
            new_prediction = true_prediction
            for i, d in enumerate(self.descriptors):
                if descriptor_vector.iloc[0,i]>0:
                    new_line = descriptor_vector.copy()
                    new_line.at[0,d] = 0
                    if self.model.predict(new_line)[0] == true_prediction:
                        w1_low = 500
                    else:
                        new_line = descriptor_vector.copy()
                        w1_low = 0
                        w1_high = new_line.iloc[0,i]
                        while w1_low != w1_high: 
                            mid = (w1_low + w1_high)//2
                            if w2_high - w2_low == 1:
                                mid = w2_high
                            new_line.at[0,d] = mid
                            new_prediction = self.model.predict(new_line)
                            if new_prediction == true_prediction:
                                w1_low = mid
                            else:
                                w1_high = mid

                    new_line = descriptor_vector.copy()
                    w2_low = new_line.iloc[0,i]
                    w2_high = new_line.iloc[0,i] + 100
                    
                    while w2_low != w2_high: 
                        mid = (w2_low + w2_high)//2
                        if w2_high - w2_low == 1:
                            mid = w2_high
                        new_line.at[0,d] = mid
                        new_prediction = self.model.predict(new_line)
                        if new_prediction == true_prediction:
                            w2_low = mid
                        else:
                            w2_high = mid
                    
                    w = 1./min(w1_low, w2_low)
                else:
                    w = 0

                if w != 0:
                    #if self.isida_like:
                    #    d = self._isida2cgrtools(d)
                    #    participating_atoms = self._full_mapping_from_descriptor(mol, d)
                    #else:
                    if "*" in d:
                        d = self._aromatize(d)
                    if self.complex:
                        mol_name, frag_smiles = d.split("::")
                        if mol_name not in self.structure_cols:
                            continue
                        if isinstance(mol[mol_name], ReactionContainer):
                            react_cgr = mol[mol_name].compose()
                            frag_cgr = self._frag2cgr(frag_smiles)
                            participating_atoms = [list(i.values()) for i in frag_cgr.get_mapping(react_cgr)]
                        else:
                            participating_atoms = [list(i.values()) for i in smiles(frag_smiles).get_mapping(mol[mol_name])]
                        participating_atoms = set(list(itertools.chain.from_iterable(participating_atoms)))
                        for a in participating_atoms:
                            atom_weights[mol[mol_name]][a] += w
                    else:
                        if isinstance(mol, ReactionContainer):
                            react_cgr = mol.compose()
                            frag_cgr = self._frag2cgr(d)
                            participating_atoms = [list(i.values()) for  i in frag_cgr.get_mapping(react_cgr)]
                        else:
                            participating_atoms = [list(i.values()) for i in smiles(d).get_mapping(mol)]
                        participating_atoms = set(list(itertools.chain.from_iterable(participating_atoms)))
                        for a in participating_atoms:
                            atom_weights[mol][a] += w
        return atom_weights
    
    def output_html(self, mol, ipython: bool = True, colorbar:bool=False):
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

        ipython: bool [optional]
            If True, an IPython-like object is returned. If False, the html code is returned

        Returns
        -------
        html: HTML object
            an HTML object containing SVG image for each structure with contributions colored
        """
        
        svgs = []
        if self.complex:
            svgs += self._draw_multiple_molecules(mol, colorbar=colorbar)         
        else:
            svgs += self._draw_one_molecule(mol, colorbar=colorbar)
        no_wrap_div = '<div style="white-space: nowrap; align-items: middle">'+'{}'*len(svgs)+'</div>'
        return HTML(no_wrap_div.format(*svgs)) if ipython else no_wrap_div.format(*svgs)

    

    def _draw_one_molecule(self, mol, contributions=None, limits:bool = False, colorbar:bool=False, 
                           external_limits:List = None):
        if contributions is None:
            contributions = self.calculate_atom_contributions(mol)
        
        if external_limits is not None:
            min_value, max_value = external_limits
            if min_value > np.min(list(contributions[mol].values())):
                print("WARNING! The minimum value of the molecule's atomic contributions is lower than the given lower limit.")
            if max_value < np.max(list(contributions[mol].values())):
                print("WARNING! The maximum value of the molecule's atomic contributions is higher than the given upper limit.")
        else:
            max_value = np.max(list(contributions[mol].values()))
            min_value = np.min(list(contributions[mol].values()))

        if limits:
            print("Min value:", min_value, ", max value:", max_value)
            
        for m in contributions.keys():
            contr = contributions[m]
            if isinstance(m, ReactionContainer):
                if self.reaction=="reactants":
                    m = self._unite_mol_list(m.reactants)
                elif self.reaction=="products":
                    m = self._unite_mol_list(m.products)
            ext_svg = m.depict()[:-6]
            ext_svg = '<svg style="background-color:white" '+ext_svg[4:]
            for k, c in contr.items():
                x, y = m.atom(k).xy[0], -m.atom(k).xy[1]
                if len(m.atom(k).atomic_symbol) >1:
                    x -= 0.1
                if self.model_type=="R":
                    color = rgb2hex(self.colormap((c+max(np.abs([min_value, max_value])))/2./max(np.abs([min_value, max_value]))))
                elif self.model_type=="C":
                    color = rgb2hex(self.colormap((c-min_value)/(max_value-min_value)))
                ext_svg += '<circle cx="{}" cy="{}" r="0.33" stroke="{}" stroke-width="0.1" fill="none" />'.format(x, y, color)
            ext_svg += "</svg>"
            if colorbar:
                w = float(ext_svg.split('"')[3][:-2])/8+1.01
                h = float(ext_svg.split('"')[3][:-2])
                cm_svg = self._colorbar_to_svg(min_value,max_value, w, h)
                return [ext_svg, cm_svg]
            return [ext_svg]

    
    def _draw_multiple_molecules(self, mol:DataFrame, limits:bool = False, colorbar:bool=False, 
                           external_limits:List = None):
        contributions = self.calculate_atom_contributions(mol)
        if external_limits is None:
            numerical_contributions = sum([list(cc.values()) for cc in contributions.values()], [])
            if self.model_type == "C":
                max_value = np.max(numerical_contributions)
                min_value = np.max(numerical_contributions)
            elif self.model_type == "R":
                max_value = np.max(np.abs(numerical_contributions))
                min_value = -max_value
        else:
            min_value, max_value = external_limits
        svgs = []
        for m in self.structure_cols[:-1]:
            svgs += self._draw_one_molecule(mol[m], contributions={mol[m]:contributions[mol[m]]}, external_limits=[min_value, max_value])
        svgs += self._draw_one_molecule(mol[self.structure_cols][-1], 
                                        contributions={mol[self.structure_cols][-1]:contributions[mol[self.structure_cols][-1]]}, 
                                        external_limits=[min_value, max_value], colorbar=True)
        return svgs

    def _plot_to_svg(self) -> str:
        s = StringIO()
        plt.savefig(s, format="svg")
        plt.close()  # https://stackoverflow.com/a/18718162/14851404
        return s.getvalue()
    
    def _colorbar_to_svg(self, min_value, max_value, width, height):
        cm = 1/2.54
        # Make a figure and axes with dimensions as desired.
        fig = plt.figure(figsize=(1, 8))
        ax = fig.add_axes([0.05, 0.15, 0.15, 0.80])
        
        if self.model_type=="C":
            norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)
            cb1 = mpl.colorbar.ColorbarBase(ax, cmap=self.colormap,
                                        norm=norm,
                                        orientation='vertical',
                                        ticks=np.linspace(min_value, max_value, 5))
        else:
            norm = mpl.colors.Normalize(vmin=-max(np.abs(min_value), np.abs(max_value)), 
                                        vmax=max(np.abs(min_value), np.abs(max_value)))
            cb1 = mpl.colorbar.ColorbarBase(ax, cmap=self.colormap,
                                        norm=norm,
                                        orientation='vertical',
                                        ticks=np.linspace(-max(np.abs(min_value), np.abs(max_value)),
                                                          max(np.abs(min_value), np.abs(max_value)),
                                                          5))
            
        cb1.ax.tick_params(labelsize="small")
        fig.set_size_inches(width*cm, height*cm/2)
    
        return self._plot_to_svg()[156:]
        
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

    def _frag2cgr(self, frag_smiles):
        if ">" in frag_smiles:
            frag1 = frag_smiles
            while frag1.find(">")>0:
                dyn_bond = frag1[frag1.index(">")-2:frag1.index(">")+3]
                frag1 = frag1.replace(dyn_bond, dyn_bond[1], 1)
            frag2 = frag_smiles
            while frag2.find(">")>0:
                dyn_bond = frag2[frag2.index(">")-2:frag2.index(">")+3]
                frag2 = frag2.replace(dyn_bond, dyn_bond[-2], 1)
            frag_cgr = ReactionContainer(reactants=[smiles(frag1)], products=[smiles(frag2)]).compose()
        else:
            frag_cgr = ReactionContainer(reactants=[smiles(frag_smiles)], products=[smiles(frag_smiles)]).compose()

        return frag_cgr

    def _unite_mol_list(self, mols):
        if len(mols)==1:
            return mols[0]
        elif len(mols)==2:
            return mols[0].union(mols[1], remap=True)
        else:
            uni = mols[0].union(mols[1], remap=True)
            for i in range(2, len(mols)):
                uni = uni.union(mols[i], remap=True)
            return uni

            
__all__ = ['ColorAtom']
