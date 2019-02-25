#!/usr/bin/env python

from distutils.core import setup
exec(compile(open('version.py', "rb").read(),
             'version.py',
             'exec'))

setup(name='ic3_labels',
      version=__version__,
      description='Creates MC labels for IceCube simulation data',
      author='Mirco Huennefeld',
      author_email='mirco.huennefeld@tu-dortmund.de',
      url='https://github.com/mhuen/ic3-labels',
      packages=['ic3_labels'],
      install_requires=['numpy', 'click', 'pyyaml',
                        ],
      )
