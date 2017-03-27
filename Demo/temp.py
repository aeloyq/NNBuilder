# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import nnbuilder
from nnbuilder.dataprepares import Load_imdb
from nnbuilder.layers import embedding,lstm,direct,logistic,softmax,encoder
from nnbuilder.algrithms import adadelta
from nnbuilder.extensions import earlystop, monitor ,sample,samples,debugmode,saveload
from nnbuilder.model import model
from nnbuilder.mainloop import train
from nnbuilder.visions.Visualization import get_result

import theano.tensor as T
import theano
theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
if __name__ == '__main__':

    global data_stream, model_stream, result_stream, vision_return

    nnbuilder.config.name='imdb'
    nnbuilder.config.max_epoches=10
    nnbuilder.config.valid_batch_size=64
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


    adadelta.config.if_clip=True

    datastream  = Load_imdb(n_words=100000,maxlen=100)
    X_mask=T.matrix('X_mask')

    emblayer=embedding.get(in_dim=100000,emb_dim=10)
    lstm_hiddenlayer=encoder.get_bi_lstm(in_dim=10,unit_dim=128)
    lstm_hiddenlayer.set_mask(X_mask)
    outputlayer=softmax.get(in_dim=128,unit_dim=2)

    model = model()
    model.X_mask=X_mask
    model.set_inputs([model.X,model.Y,X_mask])
    model.addlayer(layer=emblayer,input=model.X,name='emb')
    model.addlayer(layer=lstm_hiddenlayer,input=emblayer,name='hidden')
    model.addlayer(layer=outputlayer,input=lstm_hiddenlayer,name='output')

    result_stream = train( datastream=datastream,model=model,algrithm=adadelta, extension=[monitor])
    vision_return = get_result(result_stream=result_stream,model_stream=model)
