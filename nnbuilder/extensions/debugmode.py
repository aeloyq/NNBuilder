# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import extension
import sys
base=extension.extension
class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.kwargs=kwargs
    def init(self):
        base.init(self)
    def before_train(self):
        kwargs=self.kwargs
        train_minibatches = kwargs['minibatches'][0]
        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['data_stream']
        debug_model=kwargs['debug_model']
        kwargs['model_stream']
        data = kwargs['prepare_data'](valid_X, valid_Y, [0,1,2])
        data = tuple(data)
        kwargs['debug_result'].append(debug_model(*data))
        self.logger("Debug model finished abort trainning",1,1)
        kwargs['stop'] = True

config=ex({})