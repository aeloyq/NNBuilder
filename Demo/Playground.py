# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import sys
sys.path.append('..')
import nnbuilder as nnb
from Demo import conf
import theano

if __name__ == '__main__':
    global conf, data_stream, model_stream,result_stream,vision_return
    conf = conf.get_conf_xor()
    data_stream = conf['data_pre'](conf)
    model_stream = nnb.model.Get_Model_Stream(conf, data_stream, conf['algrithm'])
    result_stream = nnb.mainloop.train(conf, model_stream, data_stream, [nnb.extensions.monitor, nnb.extensions.earlystop])
    vision_return = nnb.visions.Visualization.get_result(result_stream, model_stream)
    '''
    conf = conf.get_conf_xor()
    data_stream = conf['data_pre'](conf)
    md=nnb.Model.model(conf)
    md.addlayer(nnb.layers.HiddenLayer.layer(conf['rng'],28*28,500),'raw','hidden500')
    md.addlayer(nnb.layers.Softmax.layer(conf['rng'],500,10),'hidden500','output')
    model_stream = nnb.Model.Get_Model_Stream(conf, data_stream,md)
    result_stream = nnb.MainLoop.Train(conf, model_stream, data_stream, nnb.algrithms.MSGD, [])
    vision_return = nnb.visions.Visualization.get_result(result_stream, model_stream)
'''