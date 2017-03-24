# -*- coding: utf-8 -*-
"""
Created on  Feb 28 9:47 PM 2017

@author: aeloyq
"""
import numpy as np
import os
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

name='unamed'
rng=np.random.RandomState(1234)
trng=RandomStreams(1234)
batch_size=20
valid_batch_size=64
max_epoches=1000
transpose_x=False
transpose_y=False
mask_x=False
mask_y=False
int_x=False
int_y=False
savelog=True