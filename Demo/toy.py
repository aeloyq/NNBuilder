# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import nnbuilder
from nnbuilder.dataprepares import Load_imdb
from nnbuilder.layers import embedding,decoder,lstm,direct,logistic,softmax,encoder,readout
from nnbuilder.algrithms import adadelta,sgd
from nnbuilder.extensions import earlystop, monitor ,sample,samples,debugmode,saveload
from nnbuilder.model import model
from nnbuilder.mainloop import train
from nnbuilder.visions.Visualization import get_result

import theano.tensor as T
import theano


nnbuilder.config.name='imdb'
nnbuilder.config.max_epoches=10
nnbuilder.config.valid_batch_size=1
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
lstm_hiddenlayer=encoder.get_bi_lstm(in_dim=500,unit_dim=1280)
lstm_hiddenlayer.set_mask(X_mask)
lstm_hiddenlayer.output_way=lstm_hiddenlayer.output_ways.mean_pooling

dec=decoder.get_rnn(1280,1280)
dec.set_y_mask(X_mask)
dec.set_x_mask(X_mask)
hidden=readout.get(1280,500)

outputlayer=softmax.get(in_dim=500,unit_dim=20000)

model = model()
model.X_mask=X_mask
model.set_inputs([model.X,model.Y,X_mask])
model.addlayer(layer=emblayer,input=model.X,name='emb')
model.addlayer(layer=lstm_hiddenlayer,input=emblayer,name='hidden')
model.addlayer(layer=dec,input=lstm_hiddenlayer,name='dec')
model.addlayer(layer=hidden,input=dec,name='rdo')
model.addlayer(layer=outputlayer,input=hidden,name='output')

result_stream = train( datastream=datastream,model=model,algrithm=sgd, extension=[monitor])
vision_return = get_result(result_stream=result_stream,model_stream=model)
