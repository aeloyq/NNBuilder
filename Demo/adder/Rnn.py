# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""

import nnbuilder
from nnbuilder.dataprepares import Load_add
from nnbuilder.layers import recurrent,direct,logistic,lstm
from nnbuilder.algrithms import adadelta,sgd
from nnbuilder.extensions import earlystop, monitor ,sample,samples,debugmode,saveload
from nnbuilder.model import model
from nnbuilder.main import train
from nnbuilder.visions.Visualization import get_result
import theano
import theano.tensor as T

if __name__ == '__main__':

    global data_stream, model_stream, result_stream, vision_return
    theano.config.NanGuardMode.nan_is_error = True
    nnbuilder.config.name='adder'
    nnbuilder.config.max_epoches=2500
    nnbuilder.config.valid_batch_size=64
    nnbuilder.config.batch_size=64
    nnbuilder.config.transpose_inputs=True

    earlystop.config.patience=10000
    earlystop.config.valid_freq=2500

    saveload.config.save_freq=2500

    sample.config.sample_func=samples.add_sample

    datastream  = Load_add()

    X=T.tensor3('X')
    Y=T.imatrix('Y')
    X_mask=T.matrix('X_mask')

    rnn_hiddenlayer=lstm.get(in_dim=10,unit_dim=10,activation=T.tanh)
    rnn_hiddenlayer.output_way=rnn_hiddenlayer.output_ways.final
    rnn_hiddenlayer.set_mask(X_mask)
    outputlayer=logistic.get(in_dim=10,unit_dim=10)
    outputlayer.cost_function=outputlayer.cost_functions.square

    model = model()
    model.X=X
    model.X_mask=X_mask
    model.Y=Y
    model.set_inputs([X,Y,X_mask],[X,Y,X_mask])
    model.addlayer(layer=rnn_hiddenlayer,input=model.X,name='hidden')
    model.addlayer(layer=outputlayer,input=rnn_hiddenlayer,name='output')

    result_stream = train( datastream=datastream,model=model,algrithm=adadelta, extension=[saveload,monitor])
    vision_return = get_result(result_stream=result_stream,model_stream=model)
