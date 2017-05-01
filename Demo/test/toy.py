# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import nnbuilder
from nnbuilder.data import Load_imdb
from nnbuilder.layers import embedding,lstm,direct,logistic,softmax,readout
from nnbuilder.models import encoder,decoder
from nnbuilder.algrithms import adadelta,sgd
from nnbuilder.extensions import earlystop, monitor ,sample,samples,debugmode,saveload
from nnbuilder.model import model
from nnbuilder.main import train
from nnbuilder.visions.Visualization import get_result

import theano.tensor as T
import theano


nnbuilder.config.name='imdb'
nnbuilder.config.max_epoches=10
nnbuilder.config.valid_batch_size=32
nnbuilder.config.batch_size=64
nnbuilder.config.transpose_x=True
nnbuilder.config.mask_x=True
nnbuilder.config.data_path='./Datasets/imdb.pkl'

earlystop.config.patience=10000
earlystop.config.valid_freq=2500
sample.config.sample_func=samples.add_sample
saveload.config.save_freq=2500
saveload.config.load=False
monitor.config.report_iter=True
monitor.config.report_iter_frequence=2



datastream  = Load_imdb(n_words=100000,maxlen=100)
X_mask=T.matrix('X_mask')

emblayer=embedding.get(in_dim=100000,emb_dim=500)
lstm_hiddenlayer=encoder.get_bi_gru__(in_dim=500,unit_dim=1280)
lstm_hiddenlayer.set_x_mask(X_mask)



outputlayer=softmax.get(in_dim=1280*2,unit_dim=2)

model = model()
model.X_mask=X_mask
model.set_inputs([model.X,model.Y,X_mask])
model.addlayer(layer=emblayer,input=model.X,name='emb')
model.addlayer(layer=lstm_hiddenlayer,input=emblayer,name='hidden')
model.addlayer(layer=outputlayer,input=lstm_hiddenlayer,name='output')

result_stream = train( datastream=datastream,model=model,algrithm=sgd, extension=[monitor])
