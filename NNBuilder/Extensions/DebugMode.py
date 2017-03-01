# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import Extension

base=Extension.extension
class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.kwargs=kwargs
    def before_train(self):
        kwargs=self.kwargs
        train_minibatches = kwargs['minibatches'][0]
        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['data_stream']
        datastream, train_model, valid_model, test_model, sample_model, debug_model, model, NNB_model, optimizer = \
        kwargs['model_stream']
        data = kwargs['prepare_data'](valid_X, valid_Y, [0,1,2])
        data = tuple(data)
        kwargs['debug_result'].append(debug_model(*data))
        print "Debug model finished abort trainning"
        kwargs['stop'] = True

config=ex({})