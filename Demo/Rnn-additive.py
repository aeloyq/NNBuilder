# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import sys
sys.path.append('..')
import NNBuilder as nnb
import numpy as np
import theano


def get_conf():
    configuration = {}
    rng = np.random.RandomState(1234)
    configuration['rng'] = rng
    # Data Preparations
    configuration['n_data']=10000
    configuration['data_pre'] = nnb.Preparation.DataPrepares.Load_add
    # MBGD Settings
    configuration['momentum_factor'] = 0.9
    configuration['max_epoches'] = 2
    configuration['learning_rate'] = 0.01
    configuration['batch_size'] = 20
    print '\r\nConfigurations:\r\n'
    for config in configuration:
        print '        ',config,':',configuration[config]
    return configuration


if __name__ == '__main__':

    global conf, data_stream, model_stream,result_stream,vision_return
    conf = get_conf()
    data_stream = data=conf['data_pre'](conf)
    conf['load_model']=nnb.Models.RNN.get_model(512,10,[10,1],10,1024)
    model_stream = nnb.Model.Get_Model_Stream(conf, data_stream,nnb.Algrithms.SGD)#,grad_monitor=True)
    result_stream = nnb.MainLoop.Train(conf, model_stream, data_stream,[nnb.Extensions.Monitor])
    vision_return = nnb.Visions.Visualization.get_result(result_stream, model_stream)