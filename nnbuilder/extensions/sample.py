# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import theano
import numpy as np
import nnbuilder.config
from extension import extension


class ex(extension):
    def __init__(self, kwargs):
        extension.__init__(self, kwargs)
        self.sample_times = 1
        self.kwargs = kwargs
        self.sample_func = None
        self.sample_freq = -1
        self.sample_from = 'train'

    def init(self):
        extension.init(self)

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
                cost = 0
                error = 0
                sr = []
                for _ in range(self.sample_times):
                    sample_data = self.get_sample_data()
                    sp_pred, sp_cost, sp_error = sample_model(*sample_data)
                    s2p = self.sample_func(sample_data[0], sp_pred, sample_data[1])
                    sr.append(s2p)
                    cost += sp_cost
                    error += sp_error
                rs = sr[0]
                for r in sr[1:]:
                    rs += "\r\n" + r
                ln = max([len(i) for i in rs.split('\r\n')])
                self.logger("Sample:", 1)
                self.logger('-' * ln, 1)
                for r in sr:
                    self.logger(r, 1)
                    self.logger('-' * ln, 1)
                self.logger("Sample Loss:%.4f   Error:%.4f%%   On Average" % (
                cost / self.sample_times, (error / self.sample_times * 100)), 1)
                self.logger('', 1)

    def get_sample_data(self):
        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = self.kwargs['datas']

        def g(X, Y):
            mask_x = None
            mask_y = None
            try:
                n_train = X.get_value().shape[0]
            except:
                n_train = len(X)
            index = nnbuilder.config.rng.randint(0, n_train)

            data_x = X
            data_y = Y
            x = [data_x[index]]

            if nnbuilder.config.transpose_x:
                x = np.asarray(x)
                x = x.transpose()
                mask_x = np.ones([x.shape[0], 1]).astype(theano.config.floatX)
            y = [data_y[index]]
            if nnbuilder.config.transpose_y:
                y = np.asarray(y)
                y = y.transpose()
                mask_y = np.ones([y.shape[0], 1]).astype(theano.config.floatX)
            if nnbuilder.config.int_x: x = np.asarray(x).astype('int64').tolist()
            if nnbuilder.config.int_y: y = np.asarray(y).astype('int64').tolist()
            data = [x, y]
            if nnbuilder.config.mask_x:
                data.append(mask_x)
            if nnbuilder.config.mask_y:
                data.append(mask_y)
            data = tuple(data)
            return data

        if self.sample_from == 'train':
            return g(train_X, train_Y)
        elif self.sample_from == 'valid':
            return g(valid_X, valid_Y)
        elif self.sample_from == 'test':
            return g(test_X, test_Y)


config = ex({})
