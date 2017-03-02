# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import sys
sys.path.append('..')
import NNBuilder
from NNBuilder.DataPrepares import Load_mnist
from NNBuilder.Algrithms import SGD,Adadelta
from NNBuilder.Extensions import Earlystop, Monitor ,Sample,Samples
from NNBuilder.Models import SoftmaxRegression
from NNBuilder.Model import Get_Model_Stream
from NNBuilder.MainLoop import Train
from NNBuilder.Visions.Visualization import get_result


if __name__ == '__main__':
    global data_stream, model_stream, result_stream, vision_return
    NNBuilder.config.data_path="./datasets/mnist.pkl.gz"
    NNBuilder.config.name='MNIST_DEMO'
    data_stream  = Load_mnist()
    model = SoftmaxRegression.get_model(n_in=28*28,hl=[500,1],n_out=10)
    model.append([(),'addreg'])
    Earlystop.config.patience=10000
    Earlystop.config.valid_freq=2500
    SGD.config.learning_rate=0.01
    model_stream = Get_Model_Stream(datastream=data_stream, algrithm= SGD, packed_model=model)  # ,grad_monitor=True)
    Sample.config.sample_func=Samples.mnist_sample
    result_stream = Train(model_stream=model_stream, datastream=data_stream, extension=[Monitor,Earlystop,Sample])
    vision_return = get_result(result_stream=result_stream, model_stream=model_stream)
