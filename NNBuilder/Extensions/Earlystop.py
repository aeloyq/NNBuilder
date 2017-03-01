# -*- coding: utf-8 -*-
"""
Created on  Feb 25 4:11 PM 2017

@author: aeloyq
"""
import Extension
import numpy as np

base=Extension.extension
class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.kwargs=kwargs
        self.imp_threshold=0.995
        self.patience_increase=2
        self.valid_freq=0
    def init(self):
        if self.valid_freq==0:
            kwargs = self.kwargs
            data = kwargs['data_stream'][0]
            try:
                n_train = data.get_value().shape[0]
            except:
                n_train = len(data)
            self.patience = n_train
            self.valid_freq=min((n_train-1)//kwargs['conf'].batch_size+1,self.patience)
    def after_iteration(self):
        kwargs=self.kwargs
        if (kwargs['iteration_total'][0]) % self.valid_freq == 0:
            valid_minibatches = kwargs['minibatches'][1]
            train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['data_stream']
            datastream, train_model, valid_model, test_model, sample_model, debug_model, model, NNB_model, optimizer = kwargs['model_stream']
            validdatas = []
            for _, index in valid_minibatches:
                data = kwargs['prepare_data'](valid_X,valid_Y, index)
                validdatas.append(data)
            valid_error = np.mean([valid_model(*tuple(validdata)) for validdata in validdatas])
            if valid_error < kwargs['best_valid_error'][0]:
                if valid_error < kwargs['best_valid_error'][0] * self.imp_threshold:
                    self.patience = max(self.patience, kwargs['iteration_total'][0] * self.patience_increase)
                kwargs['best_valid_error'][0] = valid_error
                kwargs['best_iter'][0] = kwargs['iteration_total'][0]
                print ''
                print "★Better Model Detected at Epoches:%d  Iterations:%d  Cost:%.4f  Valid error:%.4f%%★" % (
                    kwargs['epoches'][0], kwargs['iteration_total'][0], kwargs['train_result'][0], valid_error * 100)
                print ''
            if self.patience < kwargs['iteration_total'][0]:
                print "\r\n▲NO Trainning Patience      Early Stopped▲\r\n"
                kwargs['stop']=True


config=ex({})