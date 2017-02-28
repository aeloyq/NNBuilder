# -*- coding: utf-8 -*-
"""
Created on  Feb 25 4:11 PM 2017

@author: aeloyq
"""
import Extension
import numpy as np

base=Extension.extension
class ex(base):
    def __init__(self,kwargs,conf):
        base.__init__(self,kwargs,conf)
        self.kwargs=kwargs
        self.imp_threshold=0.995
        data=kwargs['data_stream'][0]
        try:
            n_train_batches = data.get_value().shape[0]
        except:
            n_train_batches = len(data)
        self.patience=n_train_batches
        self.patience_increase=2
        valid_freq=min(n_train_batches,)
    def after_iteration(self):
        kwargs=self.kwargs
        if (kwargs['iteration_total'][0]) % kwargs['valid_freq'] == 0:
            valid_minibatches = kwargs['minibatches'][1]
            train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['datastream']
            datastream, train_model, valid_model, test_model, sample_model, debug_model, model, NNB_model, optimizer = kwargs['model_stream']
            validdatas = []
            for _, index in valid_minibatches:
                x = [valid_X[t] for t in index]
                y = [valid_Y[t] for t in index]
                validdatas.append([x, y])
            valid_error = np.mean([valid_model(*tuple(validdata)) for validdata in validdatas])
            if valid_error < kwargs['best_valid_error'][0]:
                if valid_error < kwargs['best_valid_error'][0] * self.imp_threshold:
                    patience = max(self.patience, kwargs['iteration_total'][0] * self.patience_increase)
                best_valid_error = valid_error
                best_iter = kwargs['iteration_total'][0]
                print '\r\n'
                print "★Better Model Detected at Epoches:%d  Iterations:%d  Cost:%.4f  Valid error:%.4f%%★" % (
                    kwargs['epoches'][0], kwargs['iteration_total'][0], kwargs['train_cost'][0], (valid_error * 100))
                print '\r\n'
            if patience < kwargs['iteration_total'][0]:
                print "\r\n▲NO Trainning Patience      Early Stopped▲\r\n"
                dict['stop']=True