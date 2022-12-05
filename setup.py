#!/usr/bin/env python3
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
#
from pathlib import Path
from setuptools import setup, find_packages


version = '0.1.0'

setup(
    name='ChemInfoTools',
    version=version,
    packages=find_packages(),
    url='https://github.com/POSidorov/ChemInfoTools',
    license='LGPLv3',
    author=['Dr. Pavel Sidorov'],
    author_email=['pavel.o.sidorov@gmail.com'],
    python_requires='>=3.7.0',
    install_requires=['pandas>=0.25', 'numpy>=1.20', 'scipy>=1.7', 'matplotlib>=3.4', 'cgrtools>=4.1',
                      'scikit-learn>=0.24', 'ipython>=7.22'],
    long_description=(Path(__file__).parent / 'README.md').open(encoding='utf-8').read(),
    classifiers=['Environment :: Plugins',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'Topic :: Scientific/Engineering :: Chemistry',
                 'Topic :: Software Development :: Libraries :: Python Modules',
                 'License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)',
                 'Operating System :: OS Independent',
                 'Programming Language :: Python',
                 'Programming Language :: Python :: 3',
                 'Programming Language :: Python :: 3.7',
                 ]
)