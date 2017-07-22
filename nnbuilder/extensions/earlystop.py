# -*- coding: utf-8 -*-
"""
Created on  Feb 25 4:11 PM 2017

@author: aeloyq
"""
from extension import extension
from nnbuilder.main import mainloop
import numpy as np

class ex(extension):
    def __init__(self,kwargs):
        extension.__init__(self,kwargs)
        self.kwargs=kwargs
        self.imp_threshold=0.995
        self.patience_increase=2
        self.valid_freq=0
        self.valid_epoch=True
        self.patience=0
        self.valid_losss=[]
        self.valid_errors=[]
        self.best_iter=-1
        self.best_valid_error=1.
        self.save_when_valid=False
    def init(self):
        extension.init(self)
        kwargs = self.kwargs
        if self.valid_freq==0:
            n_train=np.sum(kwargs['n_data'][0])
            self.patience = n_train
            self.valid_freq=min(np.sum(kwargs['n_data'][1]),self.patience)
        else:
            n_train=np.sum(kwargs['n_data'][0])
            self.patience = n_train
    def after_iteration(self):
        if not self.valid_epoch:
            if (self.kwargs['n_iter']) % self.valid_freq == 0:
                self.valid()

    def after_epoch(self):
        if self.valid_epoch:
                self.valid()

    def valid(self):
        kwargs = self.kwargs
        valid_minibatches = kwargs['minibatches'][1]
        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['datas']
        valid_model = kwargs['valid_fn']
        validdatas = []
        for  index in valid_minibatches:
            data = mainloop.prepare_data(valid_X, valid_Y, index)
            validdatas.append(data)
        valid_error = np.mean([valid_model(*tuple(validdata)) for validdata in validdatas])
        if valid_error < self.best_valid_error:
            if valid_error < self.best_valid_error * self.imp_threshold:
                self.patience = max(self.patience, kwargs['n_iter'] * self.patience_increase)
            self.best_valid_error = valid_error
            self.best_iter = kwargs['n_iter']
            self.best_valid_error=valid_error
            self.best_iter= kwargs['n_iter']
            self.logger("Better Model Detected at Epoches:%d  Iterations:%d  Valid error:%.4f%%" % (
                kwargs['n_epoch'], kwargs['n_iter'], float(str(valid_error)) * 100), 1, 1)
        if self.patience < kwargs['n_iter']:
            self.logger("NO Trainning Patience      Early Stopped", 1, 1)
            kwargs['stop'] = True
        if self.save_when_valid:
            kwargs['saveload_save']='valid/'

    def save_(self,dict):
        dict['earlystop']={'best_iter':self.best_iter,'best_valid_error':self.best_valid_error,'valid_losss':self.valid_losss,'valid_errors':self.valid_errors}

    def load_(self,dict):
        try:
            self.best_iter=dict['earlystop']['best_iter']
            self.best_valid_error=dict['earlystop']['best_valid_error']
            self.valid_losss=dict['earlystop']['valid_losss']
            self.valid_errors=dict['earlystop']['valid_errors']
        except:
            self.logger("Didn't use earlystop in latest training Abort monitor loading", 1)



config=ex({})