# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import sys
sys.path.append('..')
import NNBuilder as nnb
from Demo import conf
import theano

if __name__ == '__main__':
    #theano.config.exception_verbosity='high'
    theano.config.profile=True
    print theano.config.profile
    global conf, data_stream, model_stream,result_stream,vision_return
    conf = conf.get_conf_xor()
    data_stream = conf['data_pre'](conf)
    model_stream = nnb.Model.Get_Model_Stream(conf, data_stream,conf['algrithm'])
    result_stream = nnb.MainLoop.Train(conf, model_stream, data_stream,[])
    vision_return = nnb.Visions.Visualization.get_result(result_stream, model_stream)
    '''
    conf = conf.get_conf_xor()
    data_stream = conf['data_pre'](conf)
    md=nnb.Model.model(conf)
    md.addlayer(nnb.Layers.HiddenLayer.layer(conf['rng'],28*28,500),'raw','hidden500')
    md.addlayer(nnb.Layers.Softmax.layer(conf['rng'],500,10),'hidden500','output')
    model_stream = nnb.Model.Get_Model_Stream(conf, data_stream,md)
    result_stream = nnb.MainLoop.Train(conf, model_stream, data_stream, nnb.Algrithms.MSGD, [])
    vision_return = nnb.Visions.Visualization.get_result(result_stream, model_stream)
'''