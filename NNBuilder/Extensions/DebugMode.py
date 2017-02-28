# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import Extension

base=Extension.extension
class ex(base):
    def __init__(self,**kwargs):
        base.__init__(self,**kwargs)
        self.kwargs=kwargs
    def before_train(self):
        kwargs=self.kwargs
        train_minibatches = kwargs['minibatches'][0]
        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['datastream']
        datastream, train_model, valid_model, test_model, sample_model, debug_model, model, NNB_model, optimizer = \
        kwargs['model_stream']
        x = [train_X[0][t] for t in train_minibatches[0]]
        y = [train_Y[t] for t in train_minibatches[0]]
        data = [x, y]
        for idx, other_data in enumerate(train_X[1:]):
            o_d = [train_X[idx][t] for t in train_minibatches[0]]
            data.extend(o_d)
        data = tuple(data)
        dict['debug_result'].append(debug_model(*data))
        print "Debug model finished abort trainning"
        dict['stop'] = True

