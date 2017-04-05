# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import extension
import timeit
import numpy as np

base=extension.extension
class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.sample_times=1
        self.kwargs=kwargs
        self.sample_func = None
        self.sample_freq=-1
    def init(self):
        base.init(self)
    def before_train(self):
        kwargs = self.kwargs
        if self.sample_freq==-1:
            self.sample_freq=len(kwargs['minibatches'][0])
    def after_iteration(self):
        kwargs = self.kwargs
        if kwargs['iteration_total'] % self.sample_freq == 0:
            if self.sample_func != None:
                datastream=kwargs['data_stream']
                sample_model=kwargs['sample_model']
                self.logger('',2)
                self.logger("☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆",2)
                self.logger('____________________________________________', 2)
                for _ in range(self.sample_times):
                    sample_data = kwargs['get_sample_data'](datastream)
                    tuple(sample_data)
                    sp_pred, sp_cost, sp_error = sample_model(*sample_data)
                    s2p,sp_y=self.sample_func(sample_data[0], sp_pred,sample_data[1])
                    self.logger( s2p,2)
                    self.logger( "Expect Result:    %s"%sp_y,2)
                    self.logger('____________________________________________',2)
                self.logger( "Sample Cost:%.4f  Sample Error:%.4f%%  " % (sp_cost, (sp_error * 100)),2)
                self.logger( "☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆",2)
                self.logger( '',2)

config=ex({})
