# ChemInfoTools

## Overview

ChemInfoTools library contains tools for working with molecules and reaction in machine learning and data analysis application.

- CircuS descriptors and the code to calculate them,
- ComplexFragmentor as a tool to concatenate descriptors of several structures into one table,
- ColorAtom implementation in Python, usable with ISIDA and CircuS desciriptors.

## Installation

activate your virtual environment with python 3.7+ , clone project and cd into the ChemInfoTools folder, then run
    
    pip install -U -e .

### Requirements

The main requirement is [CGRtools library](https://github.com/cimm-kzn/CGRtools), as calculation of CircuS descriptors and ColorAtom are based on it.
If ISIDA fragments are used, [CIMtools library](https://github.com/cimm-kzn/CIMtools) is required for the wrapper over ISIDA Fragmentor.
Calculation of CircuS descriptors is organized as a scikit-learn transformer class, therefore, pandas and scikit-learn libraries are required.

## CircuS descriptors

The descriptors are implemented using CGRtools library and its native substructure extraction functions. Their functionality is the following. The user indicates the desired lower and upper limits for the size of substructures, as the topological radius (number of bonds from a certain atom). Size of 0 means only atom itself, size of 1 – atom and all atoms directly connected to it, and so on. It should be noted that due to the way how substructure extraction is implemented in CGRtools library, the size means the number of atoms from the center, and all the bonds between selected atoms will be present, which may be slightly counterintuitive (see an example for a 5-member ring below). This is repeated for all atoms in the molecule/CGR and for all sizes from lower to upper limit to construct the fragment table.

![Demonstration of CircuS](/docs/img/circus-demo1.png)