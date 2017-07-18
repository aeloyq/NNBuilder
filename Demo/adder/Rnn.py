# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""

import nnbuilder
from nnbuilder.data import *
from nnbuilder.layers.simple import *
from nnbuilder.layers.sequential import *
from nnbuilder.algrithms import *
from nnbuilder.extensions import *
from nnbuilder.model import *
from nnbuilder.main import *


nnbuilder.config.name='adder'
nnbuilder.config.max_epoches=2500
nnbuilder.config.valid_batch_size=64
nnbuilder.config.batch_size=64
nnbuilder.config.transpose_x=True

earlystop.config.patience=10000
earlystop.config.valid_freq=2500

saveload.config.save_freq=2500

sample.config.sample_func=samples.add_sample

datastream  = Load_add()

model = model(10,Float3dX,Int2dY)
model.add(rnn(10,activation=T.nnet.sigmoid,mask=False,out='final'))
model.add(direct(cost_function=cross_entropy))

train( datastream=datastream,model=model,algrithm=sgd, extension=[monitor])

