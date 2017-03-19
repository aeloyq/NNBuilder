# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""

import nnbuilder
from nnbuilder.dataprepares import Load_add
from nnbuilder.layers import recurrent,direct
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

    earlystop.config.patience=10000
    earlystop.config.valid_freq=2500
    sample.config.sample_func=samples.add_sample
    saveload.config.save_freq=2500
    saveload.config.load=False

    adadelta.config.if_clip=True

    datastream  = Load_add()

    X=T.tensor3('X')
    Y=T.imatrix('Y')
    X_mask=T.matrix('X_mask')

    rnn_hiddenlayer=recurrent.get_new(in_dim=10,unit_dim=10)
    rnn_hiddenlayer.set_mask(X_mask)
    outputlayer=direct.get_new()

    model = model()
    model.X=X
    model.X_mask=X_mask
    model.Y=Y
    model.set_inputs([X,Y,X_mask],[X,Y,X_mask])
    model.addlayer(layer=rnn_hiddenlayer,input=model.X,name='hidden')
    model.addlayer(layer=outputlayer,input=rnn_hiddenlayer,name='output')

    result_stream = train( datastream=datastream,model=model,algrithm=adadelta, extension=[monitor,saveload])
    vision_return = get_result(result_stream=result_stream,model_stream=model)
