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
        self.report_iter=False
        self.report_epoch = True
        self.kwargs=kwargs
    def after_iteration(self):
        kwargs = self.kwargs
        if iteration_total % sample_freq == 0 and sample_freq != -1:
            if sample_func != None:
                sample_X = sample_data[0][(iteration_train_index - 1) * batch_size:(
                                                                                       iteration_train_index - 1) * batch_size + n_sample].eval()
                sample_Y = sample_data[1][(iteration_train_index - 1) * batch_size:(
                                                                                       iteration_train_index - 1) * batch_size + n_sample].eval()
                sp_pred, sp_cost, sp_error = sample_model(sample_X, sample_Y)
                sample_func(sample_X, sp_pred)
                print ''
                print "☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆"
                print "Sample Cost:%.4f  Sample Error:%.4f%%  " % (sp_cost, (sp_error * 100))
                print "☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆"
                print ''

