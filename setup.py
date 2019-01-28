"""
Copyright (C) 2019  Patrick Schwab, ETH Zurich

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from distutils.core import setup
from setuptools import find_packages

setup(
    name='ame_starter',
    version='1.0.0',
    packages=find_packages(),
    url='schwabpatrick.com',
    author='Patrick Schwab',
    author_email='patrick.schwab@hest.ethz.ch',
    license=open('LICENSE.txt').read(),
    long_description=open('README.txt').read(),
    install_requires=[
        "Keras >= 1.2.2",
        "tensorflow == 1.4.0",
        "matplotlib >= 1.3.1",
        "pandas >= 0.18.0",
        "h5py >= 2.6.0",
        "scikit-learn == 0.19.0",
        "numpy >= 1.14.1",
    ]
)
