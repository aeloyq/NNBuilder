# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import sys
sys.path.append('..')
import nnbuilder
import nnbuilder.config as config
from nnbuilder.dataprepares import Load_add
from nnbuilder.algrithms import sgd,momentum,adadelta,rmsprop
from nnbuilder.extensions import earlystop, monitor,sample,samples,debugmode
from nnbuilder.models import rnn
from nnbuilder.model import Get_Model_Stream,model
from nnbuilder.mainloop import train
from nnbuilder.visions.Visualization import get_result
from nnbuilder.layers import recurrent,direct
import theano
import theano.tensor as T


if __name__ == '__main__':
    global data_stream, model_stream, result_stream, vision_return
    config.trans_input=True
    config.max_epoches=10
    config.batch_size=128
    config.valid_batch_size=256
    sgd.config.learning_rate=0.5
    data_stream  = Load_add()
    X=T.tensor3('X')
    Y=T.imatrix('Y')
    dim_model=model()
    dim_model.set_X(X)
    dim_model.set_Y(Y)
    mask=T.matrix('mask')
    rn=recurrent.layer(config.rng, 10, 10)
    dt=direct.layer()
    dim_model.addlayer(rn,'raw','rnn',od_in=mask)
    dim_model.addlayer(dt,'rnn','direct')
    model_stream = Get_Model_Stream(datastream=data_stream, algrithm=sgd, dim_model=dim_model)  # ,grad_monitor=True)
    sample.config.sample_func=samples.add_sample
    result_stream = train(model_stream=model_stream, datastream=data_stream, extension=[monitor])
    vision_return = get_result(result_stream, model_stream)
