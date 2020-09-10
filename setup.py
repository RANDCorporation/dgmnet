#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
readme = open('README.md').read()
#history = open('HISTORY.rst').read().replace('.. :changelog:', '')
requirements = [
    'numpy',
    'pandas',
    'scikit-learn',
    'scipy',
    'matplotlib',
    'seaborn',
    'jupyter',
    'ipykernel',
    'torch',
    'networkx',
    'pyyaml'
]
#test_requirements = [
#    # TODO: put package test requirements here
#]
setup(
    name='dgmnet',
    version='1.0.0',
    description='Deep Generative Modeling for Networks',
    #long_description=readme + '\n\n' + history,
    author='Gavin Hartnett',
    author_email='hartnett@rand.org',
    url='https://code.rand.org/hartnett/dgmnet',
    #packages=[
    #    'deepwalk',
    #],
    #entry_points={'console_scripts': ['deepwalk = deepwalk.__main__:main']},
    #package_dir={'deepwalk':
    #             'deepwalk'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    #zip_safe=False,
    #keywords='deepwalk',
    #classifiers=[
    #    'Development Status :: 2 - Pre-Alpha',
    #    'Intended Audience :: Developers',
    #    'License :: OSI Approved :: BSD License',
    #    'Natural Language :: English',
    #    "Programming Language :: Python :: 2",
    #    'Programming Language :: Python :: 2.7',
    #    'Programming Language :: Python :: 3',
    #    'Programming Language :: Python :: 3.4',
    #],
    #test_suite='tests',
    #tests_require=test_requirements
)
