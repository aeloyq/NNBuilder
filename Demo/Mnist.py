# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import sys
sys.path.append('..')
import nnbuilder
from nnbuilder.dataprepares import Load_mnist
from nnbuilder.layers import hiddenlayer,softmax
from nnbuilder.algrithms import sgd
from nnbuilder.extensions import earlystop, monitor ,sample,samples,debugmode,saveload
from nnbuilder.model import model
from nnbuilder.mainloop import train
from nnbuilder.visions.Visualization import get_result

import theano
theano.config.profile=True

if __name__ == '__main__':

    global data_stream, model_stream, result_stream, vision_return

    nnbuilder.config.name= 'mnist'
    nnbuilder.config.data_path= "./datasets/mnist.pkl.gz"
    nnbuilder.config.max_epoches=10
    nnbuilder.config.valid_batch_size=20
    nnbuilder.config.batch_size=20

    earlystop.config.patience=10000
    earlystop.config.valid_freq=2500
    sgd.config.learning_rate=0.01
    sample.config.sample_func=samples.mnist_sample
    saveload.config.save_freq=2500


    datastream  = Load_mnist()

    hidden_1=hiddenlayer.get(in_dim=28*28,unit_dim=500)
    outputlayer=softmax.get(in_dim=500,unit_dim=10)

    model = model()
    model.addlayer(layer=hidden_1,input=model.X,name='hidden')
    model.addlayer(layer=outputlayer,input=hidden_1,name='output')
    #model.add_weight_decay()

    result_stream = train( datastream=datastream,model=model,algrithm=sgd, extension=[monitor])
    vision_return = get_result(result_stream=result_stream,model_stream=model)
