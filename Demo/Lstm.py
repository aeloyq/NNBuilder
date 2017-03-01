# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import sys
sys.path.append('..')
import NNBuilder
import NNBuilder.config as conf
from NNBuilder.DataPrepares import Load_add
from NNBuilder.Algrithms import SGD
from NNBuilder.Extensions import Earlystop, Monitor
from NNBuilder.Models import RNN
from NNBuilder.Model import Get_Model_Stream
from NNBuilder.MainLoop import Train
from NNBuilder.Visions.Visualization import get_result


if __name__ == '__main__':
    global data_stream, model_stream, result_stream, vision_return
    data_stream  = Load_add()
    model = RNN.get_model(n_in_vcab_length=512, n_in_word_dim=10, hl_rnn=[10, 1], n_out_word_dim=10,
                          n_out_vcab_length=1024)
    model_stream = Get_Model_Stream(data_stream, SGD, model)  # ,grad_monitor=True)
    result_stream = Train(model_stream, data_stream, [Monitor])
    vision_return = get_result(result_stream, model_stream)
