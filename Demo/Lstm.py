# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import nnbuilder
from nnbuilder.dataprepares import Load_imdb
from nnbuilder.layers import embedding,lstm,direct,logistic
from nnbuilder.algrithms import adadelta
from nnbuilder.extensions import earlystop, monitor ,sample,samples,debugmode,saveload
from nnbuilder.model import model
from nnbuilder.mainloop import train
from nnbuilder.visions.Visualization import get_result

import theano.tensor as T

if __name__ == '__main__':

    global data_stream, model_stream, result_stream, vision_return

    nnbuilder.config.name='rnn_additive'
    nnbuilder.config.max_epoches=10
    nnbuilder.config.valid_batch_size=64
    nnbuilder.config.batch_size=64
    nnbuilder.config.transpose_inputs=True
    nnbuilder.config.data_path='./Datasets/imdb.pkl'

    earlystop.config.patience=10000
    earlystop.config.valid_freq=2500
    sample.config.sample_func=samples.add_sample
    saveload.config.save_freq=2500
    saveload.config.load=False

    adadelta.config.if_clip=True

    datastream  = Load_imdb(n_words=100000,maxlen=100)
    X_mask=T.matrix('X_mask')

    emblayer=embedding.get_new(in_dim=100000,emb_dim=10)
    lstm_hiddenlayer=lstm.get_new(in_dim=10,unit_dim=10)
    lstm_hiddenlayer.set_mask(X_mask)
    lstm_hiddenlayer.output_way=lstm_hiddenlayer.output_ways.final
    outputlayer=logistic.get_new(in_dim=10,unit_dim=1)

    model = model()
    model.X_mask=X_mask
    model.set_inputs([model.X,model.Y,X_mask],[model.X,model.Y,X_mask])
    model.addlayer(layer=emblayer,input=model.X,name='emb')
    model.addlayer(layer=lstm_hiddenlayer,input=emblayer,name='hidden')
    model.addlayer(layer=outputlayer,input=lstm_hiddenlayer,name='output')

    result_stream = train( datastream=datastream,model=model,algrithm=adadelta, extension=[debugmode,monitor])
    vision_return = get_result(result_stream=result_stream,model_stream=model)
