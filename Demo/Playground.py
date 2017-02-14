# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import sys
sys.path.append('..')
import NNBuilder as nnb
from Demo import conf

if __name__ == '__main__':
    #theano.config.exception_verbosity='high'
    global conf, data_stream, model_stream,result_stream,vision_return
    conf = conf.get_conf_xor()
    data_stream = conf['data_pre'](conf)
    model_stream = nnb.Model.Get_Model_Stream(conf, data_stream,)
    result_stream = nnb.MainLoop.Train(conf, model_stream, data_stream,conf['algrithm'],[])
    vision_return = nnb.Visions.Visualization.get_result(result_stream, model_stream)
