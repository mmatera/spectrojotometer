#!/usr/bin/env python

from distutils.core import setup


def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='spectrojotometer',
      version='0.0',
      description='Toolbox for determine effective magnetic models from ab-initio simulations',
      long_description=readme(),
      classifiers=[
        'Development Status :: 1 - Alpha',
        'License :: GNU License',
        'Programming Language :: Python :: 3.4',
        'Topic :: Numerical Simulation :: Atomic Physics',
      ],
      author='Juan Mauricio Matera',
      author_email='matera@fisica.unlp.edu.ar',
      url='https://mauricio-matera.blogspot.com',
      license="GNU",
      install_requires=['argparse','numpy', 'matplotlib'],
      packages=['spectrojotometer'],
      package_data={
	'spectrojotometer': ["doc/*.html","logo.gif"],
      },
      scripts=['bin/print-equations','bin/map_configs','bin/optimize-configurations', 'bin/bond_generator','bin/evaluate_cc', 'bin/visualbond.py']
     )
