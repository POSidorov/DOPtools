DOPtools
=============

Overview
=============

DOPtools library contains tools for working with molecules and reaction in machine learning and data analysis applications.

* CircuS and Chyline descriptors and the code to calculate them,
* Calculators for fingerprints and Mordred, implemented as SKLearn compatible classes,
* ComplexFragmentor as a tool to concatenate descriptors of several structures into one table,
* ColorAtom implementation in Python, usable with CircuS and ChyLine descriptors,
* Scripts for CLI descriptor calculation and model optimization.

Installation
=============

Package can be installed from PyPI:

    pip install doptools

Otherwise, if downloaded from github, activate your virtual environment with python 3.9+ , clone project and cd into the DOPtools folder, then run

    pip install -U -e .

Requirements
============

The requirements are listed in setup.py and should be installed with the library.

The main requirement is Chython library (https://github.com/chython/chython), as calculation of molecular descriptors and ColorAtom are based on it.

Calculation of molecular descriptors is organized as a scikit-learn transformer class, therefore, pandas and scikit-learn libraries are required.

Tutorials
==================

The main functionalities of the library are demonstrated in the tutorials available in Tutorials folder.

Changelog
==================

2025-04-30 - Version 1.3.
----------------------------

Full Config for preparer. It is now possible to pass a JSON file with full configuration on the input, output and descriptor types. 
With the full config  it is possible to make any kind of combination of descriptors if you are using concatenation. The example file
is given in the examples folder. Some explanations:

The option is activated by the --full_config [filename] argument given to the launch_preparer.py script. 

.. code-block:: json-object

    "input_file": "Tutorials/Selectivity_data_full.xlsx",
    "output_folder": "output",
    "property": "ddG",
    "property_name": "ddG",

These are mandatory parameters for input and output. 

.. code-block:: json-object

    "standardize": true,
    "chiral": true,

Standardization of structures on/off, and including chirality in fingerprints on/off. 

.. code-block:: json-object

    "structures": {
        "Ar_formatted": {
            "circus": { 
                "lower":[0], 
                "upper":[2,3,4,5],
                "on_bond":[true,false]
            }
        },
        "R": {
            "circus": { 
                "lower":[0], 
                "upper":[2,3]
            },
            "chyline" : {
                "lower":[2], 
                "upper":[3,4,5]
            },
            "morgan": {
                "nBits":[1024],
                "radius":[2,3]
            }
        },
        "reaction": {
            "circus": { 
                "lower":[0], 
                "upper":[2,3]
            },
            "chyline" : {
                "lower":[2], 
                "upper":[3,4,5]
            }
        }
    },

All structural columns are now listed in this dictionary. For every column, it is possible to indicate all descriptor types and options.
The options should be given as lists, even if it is only one value. All parameters of the descriptor calculators from chem module can be used.
Be aware that the parameters such as "useFeatures" or "branchingPaths" for Morgan and RDKit FP should be given as usual, as dictionaries. 

.. code-block:: json-object

    "numerical": ["T(K)"],
    "solvent": "solvent",

"solvent" is indicating the column containing solvent names, "numernical" is for any columns that should be included in the descriptor table from the 
initial data table without change (pre-computed descriptors). 

.. code-block:: json-object

    "save": true,
    "separate_folders": false,
    "parallel": 1,
    "output_fmt": "svm"

Output parameters. Be aware that the script will currently skip the separate folder option and will output all descriptors in the same output folder.


ComplexFragmentor
==================

ComplexFragmentor class is a scikit-learn compatible transformer that concatenates the features according to specified associations. The most important argument is the *associator* - a dictionary that establishes the correspondence between a column in a data frame X and the transformer that is trained on it.

For example, say you have a data frame with molecules/CGRs in one column ("molecules"), and solvents in another ("solvent"). You want to generate a feture table that includes both structural and solvent descriptors. You would define a ComplexFragmentor class with associator as list of tuples which contain column names and the corresponding feature generators. In this case, e.g.,

    associator = [("molecules", ChythonCircus(lower=a, upper=b)), ("solvent", SolventVectorizer())] 


ComplexFragmentor assumes that at least one of the types of features will be structural, thus, *structure_columns* parameter defines the columns of the data frame where structures are found.

ColorAtom
=========

ColorAtom class implements the approach of calculating atomic contributions to the prediction by a model built using fragment descriptors. In this approach, the weights of all fragments are calculated as partial derivatives of the model’s prediction. To get the weight for one fragment, a new descriptor vector is constructed, where the value of this fragment is different (usually by value of 1 for easier calculation), the property is predicted, and the difference in predictions is taken as the weight. Each atom involved in this fragment accumulates this weight as the score, and the sum of all scores on the atom indicates its importance. This can then be visualized, by assigning colors to positive and negative colors, thus allowing to visually inspect the atomic contributions and draw conclusions which modifications to the structure may be beneficial for further improvement of the studies property.

The approach is developed and reported in 

 G. Marcou, D. Horvath, V. Solov’ev, A. Arrault, P. Vayer and A. Varnek
 Interpretability of SAR/QSAR models of any complexity by atomic contributions
 Mol. Inf., 2012, 31(9), 639-642, 2012

Current implementation is designed for both regression and classification tasks, for models built with Scikit-learn library and CircuS or ChyLine fragments implemented in chem_features module of this library. 

The application of the ColorAtom requires a trained pipeline containing a fragmentor (CircuS and ChyLine are supported), features preprocessing and a model. *calculate_atom_contributions* calculates the contributions of each atom for a given molecule and returns them numerically as a dictionary. Otherwise, they can visualized directly in Jupyter Notebook via *output_html* function that returns an HTML table containing an SVG for each structure in the molecule. Since complexFragmentor is also supported, several structures in one data point can be processed simultaneously. 

The coloring is done with matplotlib library. The atom contributions are normalized between 0 and 1 according to the maximum absolute value of the contribution. Therefore, if several structures are present, they will all have their colors normalized by the maximum value amond all contributions. The default colormap is PiYG. The "lower" (more negative) contributions are shown by red color, the "upper" (more positive) - by green.
For classification models, the coloring in monochromatic (blue), and the intensity reflects the importance of the atom (the more intense the color, the more it would affect the change in prediction if changed).

Copyright
============
2023-2025 Pavel Sidorov pavel.o.sidorov@gmail.com main developer

Contributors
============
* Philippe Gantzer p.gantzer@icredd.hokudai.ac.jp
* Iuri Casciuc yurii.kashuk@gmail.com
* Said Byadi saidbyadi@icredd.hokudai.ac.jp
* Timur Gimadiev timur.gimadiev@gmail.com
