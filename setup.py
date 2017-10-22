# -*- coding: utf-8 -*-
"""
Created on  Feb 12 11:29 AM 2017

@author: aeloyq
"""
import setuptools
from setuptools import setup, find_packages

setup(
    name='NNBuilder',
    version='0.3.1',
    description='A multi-backend framework for building and training neural networks',
    url='https://github.com/aeloyq',
    author='aeloyq IOBLAB-Shanghai Maritime University',
    author_email='aeloyq@icloud.com',
    license='aeloyq',
    packages=find_packages(exclude=['*.pyc', 'demo', 'doc', '.idea']),
    classifiers=[
        'Programming Language :: Python :: 2.7'
    ],
    keywords='theano pytorch machine learning neural networks deep learning',
    setup_requires=['numpy'],
    install_requires=['numpy', 'theano'],
    zip_safe=False,
    conda_buildnum=1,
    conda_features=['mkl'],
)
