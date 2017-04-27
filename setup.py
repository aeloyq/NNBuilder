# -*- coding: utf-8 -*-
"""
Created on  Feb 12 11:29 AM 2017

@author: aeloyq
"""
from setuptools import setup

setup(
    name='NNBuilder',
    version='0.1.0',
    description='A Theano framework for building and training neural networks',
    url='https://github.com/aeloyq',
    author='aeloyq IOBLAB-Shanghai Maritime University',
    license='aeloyq',
    classifiers=[
        'Programming Language :: Python :: 2.7'
    ],
    keywords='theano machine learning neural networks deep learning',
    setup_requires=['numpy'],
    install_requires=['numpy', 'six', 'pyyaml', 'toolz', 'theano',
                      'picklable-itertools', 'progressbar2', 'fuel', 'pydot', 'graphviz'],
    zip_safe=False)
