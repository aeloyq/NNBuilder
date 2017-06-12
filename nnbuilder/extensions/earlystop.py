# -*- coding: utf-8 -*-
"""
Created on  Feb 25 4:11 PM 2017

@author: aeloyq
"""
import extension
import numpy as np

base=extension.extension
class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.kwargs=kwargs
        self.imp_threshold=0.995
        self.patience_increase=2
        self.valid_freq=0
        self.valid_epoch=True
    def init(self):
        base.init(self)
        if self.valid_freq==0:
            kwargs = self.kwargs
            data = kwargs['data_stream'][0]
            try:
                n_train = data.get_value().shape[0]
            except:
                n_train = len(data)
            self.patience = n_train
            self.valid_freq=min((n_train-1)//kwargs['conf'].batch_size+1,self.patience)
        else:
            kwargs = self.kwargs
            data = kwargs['data_stream'][0]
            try:
                n_train = data.get_value().shape[0]
            except:
                n_train = len(data)
            self.patience = n_train
    def after_iteration(self):
        if not self.valid_epoch:
            kwargs=self.kwargs
            if (kwargs['iteration_total']) % self.valid_freq == 0:
                valid_minibatches = kwargs['minibatches'][1]
                train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['data_stream']
                valid_model=kwargs['valid_model']
                validdatas = []
                for _, index in valid_minibatches:
                    data = kwargs['prepare_data'](valid_X,valid_Y, index)
                    validdatas.append(data)
                valid_error = np.mean([valid_model(*tuple(validdata)) for validdata in validdatas])
                if valid_error < kwargs['best_valid_error']:
                    if valid_error < kwargs['best_valid_error'] * self.imp_threshold:
                        self.patience = max(self.patience, kwargs['iteration_total'] * self.patience_increase)
                    kwargs['best_valid_error'] = valid_error
                    kwargs['best_iter'] = kwargs['iteration_total']
                    self.logger("★Better Model Detected at Epoches:%d  Iterations:%d  Valid error:%.4f%%★" % (
                        kwargs['epoches'], kwargs['iteration_total'], valid_error * 100),2,1)
                if self.patience < kwargs['iteration_total']:
                    self.logger( "▲NO Trainning Patience      Early Stopped▲",1,1)
                    kwargs['stop']=True

    def after_epoch(self):
        if self.valid_epoch:
            kwargs = self.kwargs
            valid_minibatches = kwargs['minibatches'][1]
            train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['data_stream']
            valid_model = kwargs['valid_model']
            validdatas = []
            for _, index in valid_minibatches:
                data = kwargs['prepare_data'](valid_X, valid_Y, index)
                validdatas.append(data)
            valid_error = np.mean([valid_model(*tuple(validdata)) for validdata in validdatas])
            if valid_error < kwargs['best_valid_error']:
                if valid_error < kwargs['best_valid_error'] * self.imp_threshold:
                    self.patience = max(self.patience, kwargs['iteration_total'] * self.patience_increase)
                kwargs['best_valid_error'] = valid_error
                kwargs['best_iter'] = kwargs['iteration_total']
                self.logger("★Better Model Detected at Epoches:%d  Iterations:%d  Valid error:%.4f%%★" % (
                    kwargs['epoches'], kwargs['iteration_total'], valid_error * 100), 2, 1)
            if self.patience < kwargs['iteration_total']:
                self.logger("▲NO Trainning Patience      Early Stopped▲", 1, 1)
                kwargs['stop'] = True


config=ex({})