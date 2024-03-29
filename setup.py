#!/usr/bin/env python
import os
from setuptools import setup, find_packages

here = os.path.dirname(__file__)

about = {}
with open(os.path.join(here, 'ic3_labels', '__about__.py')) as fobj:
    exec(fobj.read(), about)

setup(
    name='ic3_labels',
    version=about['__version__'],
    packages=find_packages(),
    install_requires=[
        'numpy', 'click', 'pyyaml', 'scipy', 'MCEq', 'crflux',
        'nuVeto',
    ],
    include_package_data=True,
    author=about['__author__'],
    author_email=about['__author_email__'],
    maintainer=about['__author__'],
    maintainer_email=about['__author_email__'],
    description=about['__description__'],
    url=about['__url__']
)
