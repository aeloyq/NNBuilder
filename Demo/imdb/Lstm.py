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


nnbuilder.config.name='imdb'
nnbuilder.config.max_epoches=10
nnbuilder.config.valid_batch_size=64
nnbuilder.config.batch_size=64
nnbuilder.config.transpose_x=True
nnbuilder.config.int_x=True
nnbuilder.config.mask_x=True
nnbuilder.config.data_path='../Datasets/imdb.pkl'

earlystop.config.patience=10000
earlystop.config.valid_freq=2500
sample.config.sample_func=samples.add_sample
saveload.config.save_freq=2500
monitor.config.report_iter=True
monitor.config.report_iter_frequence=2


datastream  = Load_imdb(n_words=100000,maxlen=100)

model = model(100000,Int2dX)
model.sequential()
model.add(embedding(10))
model.add(gru(128,out='final'))
model.add(dropout(0.5))
model.add(softmax(2))


result_stream = train(datastream=datastream,model=model,algrithm=adam, extension=[monitor])
