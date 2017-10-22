# -*- coding: utf-8 -*-
"""
Created on  Feb 25 4:11 PM 2017

@author: aeloyq
"""
from basic import base
from nnbuilder.main import mainloop
import numpy as np
import saveload


class ex(base):
    name = 'earlystop'

    def __init__(self, kwargs):
        base.__init__(self, kwargs)
        self.kwargs = kwargs
        self.patience = 8
        self.valid_freq = 0
        self.valid_epoch = True
        self.valid_by_loss = False
        self.save_when_valid = False
        self.save_best_valid = True

    def init(self):
        self.valid_losses = []
        self.valid_errors = []
        self.best_iter = -1
        self.best_epoch = -1
        self.best_valid_loss = 1e8
        self.best_valid_error = 1.
        self.bad_count = 0
        base.init(self)
        kwargs = self.kwargs
        if self.valid_freq == 0:
            n_train = np.sum(kwargs['n_data'][0])
            self.valid_freq = min(np.sum(kwargs['n_data'][1]), n_train)
        if self.valid_epoch:
            self.valid_freq = np.sum(kwargs['n_data'][0])

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
        valid_fn = kwargs['valid_fn']
        validdatas = []
        for index in valid_minibatches:
            data = mainloop.prepare_data(valid_X, valid_Y, index, kwargs['model'])
            validdatas.append(data)
        valid_loss, valid_error = np.mean(np.asarray([valid_fn(*tuple(validdata)) for validdata in validdatas]), 0)
        self.valid_losses.append(valid_loss)
        self.valid_errors.append(valid_error)
        self.logger('', 1)
        self.logger('-' * 60, 1)

        def better_countered(valid_loss, valid_error):
            self.bad_count = 0
            self.best_valid_loss = valid_loss
            self.best_valid_error = valid_error
            self.best_iter = kwargs['n_iter']
            self.best_epoch = kwargs['n_epoch']
            self.logger("Better Model Detected", 1, 0)
            self.logger("Valid at Epoch:%d  Iter:%d  Loss:%.4f  Error:%.4f%%" % (
                kwargs['n_epoch'], kwargs['n_iter'], valid_loss, valid_error * 100), 1, 0)
            if self.save_best_valid:
                saveload.instance.save_npz(kwargs['model'], 'valid/best/', str(kwargs['n_iter']) + '.npz', self.kwargs)

        def bad_countered(valid_loss, valid_error):
            self.bad_count += 1
            self.logger("Patience Remain:%d   Bad Count:%d" % (self.patience - self.bad_count, self.bad_count), 1, 0)
            self.logger("Valid at Epoch:%d  Iter:%d  Loss:%.4f  Error:%.4f%%" % (
                kwargs['n_epoch'], kwargs['n_iter'], valid_loss, valid_error * 100), 1, 0)

        if not self.valid_by_loss:
            if valid_error <= self.best_valid_error:
                better_countered(valid_loss, valid_error)
            else:
                bad_countered(valid_loss, valid_error)
        else:
            if valid_loss <= self.best_valid_loss:
                better_countered(valid_loss, valid_error)
            else:
                bad_countered(valid_loss, valid_error)
        self.logger('-' * 60, 1)
        self.logger('', 1)
        if self.patience == self.bad_count:
            self.logger("")
            self.logger("NO Trainning Patience      Early Stop Soon", 1)
            self.logger("Best Valid Model at Epoch:%d at Iter:%d  Best Loss:%.4f  Best Error:%.4f%%" % (
                self.best_epoch, self.best_iter, self.best_valid_loss, self.best_valid_error * 100), 1)
            self.logger("")
            kwargs['stop'] = True
        if self.save_when_valid:
            saveload.instance.save_npz(kwargs['model'], 'valid/', kwargs['n_iter'], self.kwargs, delete=False)

    def save_(self, dict):
        dict = {'patience': self.patience, 'best_iter': self.best_iter,
                'best_valid_error': self.best_valid_error, 'valid_losses': self.valid_losses, 'valid_errors': self.valid_errors}
        return dict

    def load_(self, dict):
        try:
            self.patience = dict['patience']
            self.best_iter = dict['best_iter']
            self.best_valid_error = dict['best_valid_error']
            self.valid_losses = dict['valid_losses']
            self.valid_errors = dict['valid_errors']
        except:
            self.logger("Didn't use earlystop in latest training Abort earlystop loading", 2)


config = ex({})
instance = config
