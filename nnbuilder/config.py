# -*- coding: utf-8 -*-
"""
Created on  Feb 28 9:47 PM 2017

@author: aeloyq
"""
import numpy as np
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

name='unamed'



batch_size=20
valid_batch_size=64
max_epoch=1000
data_path=''
transpose_x=False
transpose_y=False
mask_x=False
mask_y=False
int_x=False
int_y=False
savelog=True


rng=np.random.RandomState(1234)
trng=RandomStreams(1234)