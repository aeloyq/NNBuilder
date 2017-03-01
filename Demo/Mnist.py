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
    data_stream  = Load_mnist()
    model = SoftmaxRegression.get_model(n_in=28*28,hl=[500,1],n_out=10)
    model_stream = Get_Model_Stream(datastream=data_stream, algrithm= Adadelta, packed_model=model)  # ,grad_monitor=True)
    Sample.config.sample_func=Samples.mnist_sample
    result_stream = Train(model_stream, data_stream, [Monitor,Earlystop,Sample])
    vision_return = get_result(result_stream, model_stream)
