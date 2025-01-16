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

from itertools import compress
from pandas import DataFrame
from sklearn.base import BaseEstimator, TransformerMixin


class SolventVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, sp: bool = True, sdp: bool = True, sa: bool = True, sb: bool = True): 
        self.sp = sp
        self.sdp = sdp
        self.sa = sa
        self.sb = sb
        header = []
        index = []
        if self.sp:
            header.append('SP Katalan')
            index.append(True)
        else:
            index.append(False)
        if self.sdp:
            header.append('SdP Katalan')
            index.append(True)
        else:
            index.append(False)
        if self.sa:
            header.append('SA Katalan')
            index.append(True)
        else:
            index.append(False)
        if self.sb:
            header.append('SB Katalan')
            index.append(True)
        else:
            index.append(False)
        self.__header = header
        self.__index = index
        
    def get_feature_names(self):
        return self.__header

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return DataFrame([compress(available_solvents[x], self.__index) if type(x) is str else compress((0,0,0,0), self.__index) for x in x], columns=self.__header)


available_solvents = { 
     float('nan'): (0, 0, 0, 0),
     None: (0, 0, 0, 0),
     "gas phase": (0, 0, 0, 0),
     "water": (0.681, 0.997, 1.062, 0.025),
     "tetrachloromethane": (0.768, 0, 0, 0.044),
     "carbon disulfide": (1, 0, 0, 0.104),
     "chloroform": (0.783, 0.614, 0.047, 0.071),
     "dichloromethane": (0.761, 0.769, 0.04, 0.178),
     "formamide": (0.814, 1.006, 0.549, 0.414),
     "nitromethane": (0.71, 0.954, 0.078, 0.236),
     "methanol": (0.608, 0.904, 0.605, 0.545),
     "1,1,2-trichlorotrifluoroethane": (0.596, 0.152, 0, 0.038),
     "chloroacetonitrile": (0.763, 1.024, 0.445, 0.184),
     "1,1,2,2-tetrachloroethane": (0.845, 0.792, 0, 0.017),
     "1,1,1-trichloroethane": (0.737, 0.5, 0, 0.085),
     "2,2,2-trifluoroethanol": (0.543, 0.922, 0.893, 0.107),
     "acetonitrile": (0.645, 0.974, 0.044, 0.286),
     "1,2-dichloroethane": (0.771, 0.742, 0.03, 0.126),
     "acetic acid": (0.651, 0.676, 0.689, 0.39),
     "nitroethane": (0.706, 0.902, 0, 0.234),
     "ethanol": (0.633, 0.783, 0.4, 0.658),
     "dimethyl sulfoxide": (0.83, 1, 0.072, 0.647),
     "1,2-ethanediol": (0.777, 0.91, 0.717, 0.534),
     "propanenitrile": (0.668, 0.888, 0.03, 0.365),
     "acetone": (0.651, 0.907, 0, 0.475),
     "allyl alcohol": (0.705, 0.839, 0.415, 0.585),
     "ethyl formate": (0.648, 0.707, 0, 0.477),
     "methyl acetate": (0.645, 0.637, 0, 0.527),
     "propanoic acid": (0.664, 0.434, 0.608, 0.377),
     "dimethyl carbonate": (0.653, 0.531, 0.064, 0.433),
     "N,N-dimethylformamide": (0.759, 0.977, 0.031, 0.613),
     "1-propanol": (0.658, 0.748, 0.367, 0.782),
     "2-propanol": (0.633, 0.808, 0.283, 0.83),
     "1,2-propanediol": (0.731, 0.888, 0.475, 0.598),
     "1,2,3-propanetriol": (0.828, 0.921, 0.653, 0.309),
     "trimethyl phosphate": (0.707, 0.909, 0, 0.522),
     "1-methylimidazole": (0.834, 0.959, 0.069, 0.658),
     "g-butyrolactone": (0.775, 0.945, 0.057, 0.399),
     "propylene carbonate": (0.746, 0.942, 0.106, 0.341),
     "butanenitrile": (0.689, 0.864, 0, 0.384),
     "2-butanone": (0.669, 0.872, 0, 0.52),
     "tetrahydrofuran": (0.714, 0.634, 0, 0.591),
     "butanoic acid": (0.675, 0.333, 0.571, 0.464),
     "1,4-dioxane": (0.737, 0.312, 0, 0.444),
     "ethyl acetate": (0.656, 0.603, 0, 0.542),
     "propyl formate": (0.667, 0.633, 0, 0.549),
     "2-methylpropionic acid": (0.659, 0.302, 0.515, 0.281),
     "sulfolane": (0.83, 0.896, 0.052, 0.365),
     "1-chlorobutane": (0.693, 0.529, 0, 0.138),
     "butylamine": (0.69, 0.296, 0, 0.944),
     "1-butanol": (0.674, 0.655, 0.341, 0.809),
     "2-butanol": (0.656, 0.706, 0.221, 0.888),
     "diethyl ether": (0.617, 0.385, 0, 0.562),
     "2-methyl-1-propanol": (0.657, 0.684, 0.311, 0.828),
     "2-methyl-2-propanol": (0.632, 0.732, 0.145, 0.928),
     "1,2-butanediol": (0.724, 0.817, 0.466, 0.668),
     "1,3-butanediol": (0.739, 0.873, 0.429, 0.61),
     "1,4-butanediol": (0.763, 0.864, 0.424, 0.598),
     "2,3-butanediol": (0.714, 0.877, 0.461, 0.652),
     "1,2-dimethoxyethane": (0.68, 0.625, 0, 0.636),
     "pyridine": (0.842, 0.761, 0.033, 0.581),
     "pentanenitrile": (0.696, 0.853, 0, 0.408),
     "1-methyl-2pyrrolidinone": (0.812, 0.959, 0.024, 0.613),
     "cyclopentane": (0.655, 0, 0, 0.063),
     "cyclopentanol": (0.74, 0.673, 0.258, 0.836),
     "2-methyltetrahydrofuran": (0.7, 0.768, 0, 0.584),
     "2-pentanone": (0.689, 0.783, 0.01, 0.537),
     "3-pentanone": (0.692, 0.785, 0, 0.557),
     "2-methylbutyric acid": (0.686, 0.261, 0.439, 0.25),
     "3-methylbutyric acid": (0.67, 0.31, 0.538, 0.405),
     "pentanoic acid": (0.687, 0.276, 0.502, 0.473),
     "propyl acetate": (0.67, 0.559, 0, 0.548),
     "1-methylpyrrolidine": (0.712, 0.216, 0, 0.918),
     "piperidine": (0.754, 0.365, 0, 0.933),
     "N,N-diethylformamide": (0.745, 0.939, 0, 0.614),
     "2-methylbutane": (0.581, 0, 0, 0.053),
     "pentane": (0.593, 0, 0, 0.073),
     "butyl methyl ether": (0.647, 0.354, 0, 0.505),
     "tert-butyl methyl ether": (0.622, 0.422, 0, 0.567),
     "1-pentanol": (0.687, 0.587, 0.319, 0.86),
     "2-pentanol": (0.667, 0.665, 0.204, 0.916),
     "1,1,3,3-tetramethylguanidine": (0.78, 0.75, 0, 1),
     "tetramethylurea": (0.778, 0.878, 0, 0.624),
     "hexafluorobenzene": (0.623, 0.252, 0, 0.119),
     "perfluorohexane": (0.339, 0, 0, 0.057),
     "1,2-dichlorobenzene": (0.869, 0.676, 0.033, 0.144),
     "bromobenzene": (0.875, 0.497, 0, 0.192),
     "chlorobenzene": (0.833, 0.537, 0, 0.182),
     "fluorobenzene": (0.761, 0.511, 0, 0.113),
     "nitrobenzene": (0.891, 0.873, 0.056, 0.24),
     "benzene": (0.793, 0.27, 0, 0.124),
     "aniline": (0.924, 0.956, 0.132, 0.264),
     "N,N-dimethylacetamide": (0.763, 0.987, 0.028, 0.65),
     "cyclohexanone": (0.766, 0.745, 0, 0.0482),
     "cyclohexane": (0.683, 0, 0, 0.073),
     "butyl acetate": (0.674, 0.535, 0, 0.525),
     "hexanoic acid": (0.698, 0.245, 0.465, 0.304),
     "1-methylpiperidine": (0.708, 0.116, 0, 0.836),
     "N,N-diethylacetamide": (0.748, 0.918, 0, 0.66),
     "hexane": (0.616, 0, 0, 0.056),
     "3-methylpentane": (0.62, 0, 0, 0.05),
     "1,4-dimethylpiperazine": (0.73, 0.211, 0, 0.832),
     "diisopropyl ether": (0.625, 0.324, 0, 0.657),
     "dipropyl ether": (0.645, 0.286, 0, 0.666),
     "1-hexanol": (0.698, 0.552, 0.315, 0.879),
     "2-hexanol": (0.683, 0.601, 0.14, 0.966),
     "triethylamine": (0.66, 0.108, 0, 0.885),
     "hexamethylphosphoramine": (0.744, 1.1, 0, 0.813),
     "trifluoromethylbenzene": (0.694, 0.663, 0.014, 0.073),
     "benzonitrile": (0.851, 0.852, 0.047, 0.281),
     "toluene": (0.782, 0.284, 0, 0.0128),
     "anisole": (0.82, 0.543, 0.084, 0.299),
     "benzyl alcohol": (0.861, 0.788, 0.409, 0.461),
     "cycloheptane": (0.703, 0, 0, 0.05),
     "methylcyclohexane": (0.675, 0, 0, 0.069),
     "cycloheptanol": (0.77, 0.546, 0.183, 0.911),
     "heptanoic acid": (0.704, 0.238, 0.445, 0.328),
     "heptane": (0.635, 0, 0, 0.083),
     "1-heptanol": (0.706, 0.499, 0.302, 0.912),
     "acetophenone": (0.848, 0.808, 0.044, 0.365),
     "methyl benzoate": (0.824, 0.654, 0, 0.378),
     "methyl salicylate": (0.843, 0.741, 0.219, 0.216),
     "o-xylene": (0.791, 0.266, 0, 0.157),
     "m-xylene": (0.771, 0.205, 0, 0.162),
     "p-xylene": (0.778, 0.175, 0, 0.16),
     "ethylbenzene": (0.81, 0.669, 0, 0.295),
     "ethoxybenzene": (0.81, 0.669, 0, 0.295),
     "2-phenylethanol": (0.849, 0.793, 0.376, 0.523),
     "veratrole": (0.851, 0.606, 0, 0.34),
     "cyclooctane": (0.719, 0, 0, 0.068),
     "cyclooctanol": (0.777, 0.537, 0.137, 0.919),
     "octane": (0.65, 0, 0, 0.079),
     "2,2,4-trimethylpentane": (0.618, 0, 0, 0.044),
     "dibutyl ether": (0.672, 0.175, 0, 0.637),
     "1-octanol": (0.713, 0.454, 0.299, 0.923),
     "2-octanol": (0.696, 0.496, 0.088, 0.963),
     "dibutylamine": (0.692, 0.209, 0, 0.991),
     "ethyl salicylate": (0.819, 0.681, 0.118, 0.236),
     "propylbenzene": (0.767, 0.209, 0, 0.144),
     "mesitylene": (0.775, 0.155, 0, 0.19),
     "triacetin": (0.735, 0.686, 0.023, 0.416),
     "nonane": (0.66, 0, 0, 0.053),
     "1-nonanol": (0.717, 0.429, 0.27, 0.906),
     "1-bromonaphthalene": (0.964, 0.669, 0, 0.202),
     "1,2,3,4-tetrahydronaphthalene": (0.838, 0.182, 0, 0.18),
     "butylbenzene": (0.765, 0.176, 0, 0.149),
     "tert-butylbenzene": (0.757, 0.182, 0, 0.171),
     "1,2,3,5-tetramethylbenzene": (0.784, 0.203, 0, 0.186),
     "decalin": (0.744, 0, 0, 0.056),
     "cis-decalin": (0.753, 0, 0, 0.056),
     "decane": (0.669, 0, 0, 0.066),
     "1-decanol": (0.722, 0.383, 0.259, 0.912),
     "eucalyptol": (0.736, 0.343, 0, 0.737),
     "1-methylnaphthalene": (0.908, 0.51, 0, 0.156),
     "undecane": (0.678, 0, 0, 0.08),
     "1-undecanol": (0.728, 0.342, 0.257, 0.909),
     "dodecane": (0.683, 0, 0, 0.086),
     "dihexyl ether": (0.681, 0.208, 0, 0.618),
     "tributylamine": (0.689, 0.06, 0, 0.854),
     "tridecane": (0.69, 0, 0, 0.05),
     "dibenzyl ether": (0.877, 0.509, 0, 0.33),
     "tetradecane": (0.696, 0, 0, 0.05),
     "pentadecane": (0.7,0, 0, 0.068),
     "hexadecane": (0.704, 0, 0, 0.086),
     "squalane": (0.714, 0, 0, 0.05),
     "petroleum ether": (0.593, 0.005, 0, 0.043),
}

__all__ = ['available_solvents', 'SolventVectorizer']
