# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import Extension
import timeit
import numpy as np

base=Extension.extension
class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.sample_times=1
        self.kwargs=kwargs
        self.sample_func = None
    def before_train(self):
        kwargs = self.kwargs
        self.sample_freq=len(kwargs['minibatches'][0])
    def after_iteration(self):
        kwargs = self.kwargs
        if kwargs['iteration_total'][0] % self.sample_freq == 0:
            if self.sample_func != None:
                datastream, train_model, valid_model, test_model, sample_model, debug_model, model, NNB_model, optimizer =kwargs['model_stream']
                sample_data=kwargs['get_sample_data'](datastream)
                tuple(sample_data)
                sp_pred, sp_cost, sp_error = sample_model(*sample_data)
                print ''
                print "☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆"
                for _ in range(self.sample_times):
                    s2p,sp_y=self.sample_func(sample_data[0], sp_pred,sample_data[1])
                    print s2p
                    print "Expect Result:    %s"%sp_y
                print "Sample Cost:%.4f  Sample Error:%.4f%%  " % (sp_cost, (sp_error * 100))
                print "☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆"
                print ''

config=ex({})
