# -*- coding: utf-8 -*-
"""
Created on  Feb 12 11:29 AM 2017

@author: aeloyq
"""
from setuptools import setup , find_packages

setup(
    name='NNBuilder',
    version='0.2.0',
    description='A Theano framework for building and training neural networks',
    url='https://github.com/aeloyq',
    author='aeloyq IOBLAB-Shanghai Maritime University',
    license='aeloyq',
    packages = find_packages(exclude=['*.pyc','demo','doc','.idea']),
    classifiers=[
        'Programming Language :: Python :: 2.7'
    ],
    keywords='theano machine learning neural networks deep learning',
    setup_requires=['numpy'],
    install_requires=['numpy',  'theano', 'pydot', 'pydot-ng'],
    zip_safe=False)
