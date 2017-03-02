# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import sys
sys.path.append('..')
import NNBuilder
import NNBuilder.config as config
from NNBuilder.DataPrepares import Load_add
from NNBuilder.Algrithms import SGD,Momentum,Adadelta
from NNBuilder.Extensions import Earlystop, Monitor,Sample,Samples,DebugMode
from NNBuilder.Models import RNN
from NNBuilder.Model import Get_Model_Stream
from NNBuilder.MainLoop import Train
from NNBuilder.Visions.Visualization import get_result


if __name__ == '__main__':
    global data_stream, model_stream, result_stream, vision_return
    config.trans_input=True
    config.max_epoches=1000
    SGD.config.learning_rate=1
    data_stream  = Load_add()
    model = RNN.get_model(n_in_word_dim=10, hl_rnn=[10, 1], n_out_word_dim=10)
    model_stream = Get_Model_Stream(datastream=data_stream, algrithm=Adadelta, packed_model=model)  # ,grad_monitor=True)
    Sample.config.sample_func=Samples.add_sample
    result_stream = Train(model_stream=model_stream, datastream=data_stream,extension=[Monitor,Sample])
    vision_return = get_result(result_stream, model_stream)
