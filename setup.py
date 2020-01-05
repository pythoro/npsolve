# -*- coding: utf-8 -*-
"""
Created on Wed Jul 19 15:02:12 2019

@author: Reuben
"""

import setuptools

from os import path
this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setuptools.setup(
    name="npsolve",
    version="0.0.4",
    author="Reuben Rusk",
    author_email="pythoro@mindquip.com",
    description="Easier object-oriented calculations for numerical solvers.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pythoro/npsolve.git",
    download_url="https://github.com/pythoro/npsolve/archive/v0.0.4.zip",
    packages=['npsolve'],
    keywords=['NUMERICAL SOLVER', 'NUMPY', 'SCIPY', 'ODE', 'INTEGRATION'],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Topic :: Software Development :: Libraries :: Python Modules"
    ],
    install_requires=['numpy', 'fastwire'],
)