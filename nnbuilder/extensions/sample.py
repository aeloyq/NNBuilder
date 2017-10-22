# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import numpy as np
from basic import base
from nnbuilder.main import mainloop
from nnbuilder.kernel import *


class ex(base):
    def __init__(self, kwargs):
        base.__init__(self, kwargs)
        self.sample_times = 1
        self.kwargs = kwargs
        self.sample_func = None
        self.sample_freq = -1
        self.sample_from = 'train'

    def init(self):
        base.init(self)

    def before_train(self):
        kwargs = self.kwargs
        if self.sample_freq == -1:
            self.sample_freq = kwargs['n_data'][1][kwargs['n_part']]

    def after_iteration(self):
        kwargs = self.kwargs
        if kwargs['n_iter'] % self.sample_freq == 0:
            if self.sample_func != None:
                sample_model = kwargs['sample_fn']
                self.logger('', 1)
                loss = 0
                sr = []
                for _ in range(self.sample_times):
                    sample_data = self.get_sample_data()
                    sp_pred, sp_loss = sample_model(*sample_data)
                    s2p = self.sample_func(sample_data[0], sp_pred, sample_data[1])
                    sr.append(s2p)
                    loss += sp_loss
                rs = sr[0]
                for r in sr[1:]:
                    rs += "\r\n" + r
                ln = max([len(i) for i in rs.split('\r\n')])
                self.logger("Sample:", 1)
                self.logger('-' * ln, 1)
                for r in sr:
                    self.logger(r, 1)
                    self.logger('-' * ln, 1)
                self.logger("Loss:%.4f" % (
                    loss / self.sample_times), 1)
                self.logger('', 1)

    def get_sample_data(self):
        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = self.kwargs['datas']
        if self.sample_from == 'train':
            n = len(train_X)
            index = kernel.random.randint(0, n)
            return mainloop.prepare_data(train_X, train_Y, [index], self.kwargs['model'])
        elif self.sample_from == 'valid':
            n = len(valid_X)
            index = kernel.random.randint(0, n)
            return mainloop.prepare_data(valid_X, valid_Y, [index], self.kwargs['model'])
        elif self.sample_from == 'test':
            n = len(test_X)
            index = kernel.random.randint(0, n)
            return mainloop.prepare_data(test_X, test_Y, [index], self.kwargs['model'])


config = ex({})
instance = config
