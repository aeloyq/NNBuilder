# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
from basic import *


class Sample(ExtensionBase):
    def __init__(self):
        ExtensionBase.__init__(self)
        self.num = 1
        self.freq = None
        self.callback = None
        self.sample_from = 'train'

    def before_train(self):
        if self.freq == None:
            self.freq = (self.data.size - 1) // self.config.batch_size + 1

    def after_iteration(self):
        if self.train_history['n_iter'] % self.freq == 0:
            if self.callback != None:
                self.logger('', 1)
                loss = 0
                sample_logs = []
                for _ in range(self.num):
                    sample_X, sample_Y = self.get_sample_data()
                    sample_results = self.model.sampler(sample_X, sample_Y)
                    sample_log = self.callback(sample_X[0], sample_Y[0], sample_results['predict'][0], sample_results)
                    sample_logs.append(sample_log)
                    loss += sample_results['sample_loss']
                sample_logs_str = sample_logs[0]
                for sample_log in sample_logs[1:]:
                    sample_logs_str += "\r\n" + sample_log
                length = max([len(i) for i in sample_logs_str.split('\r\n')])
                self.logger("Sample:", 1)
                self.logger('-' * length, 1)
                for sample_log in sample_logs:
                    self.logger(sample_log, 1)
                    self.logger('-' * length, 1)
                self.logger("Loss:%.4f" % (
                    loss / self.num), 1)
                self.logger('', 1)

    def get_sample_data(self):
        if self.sample_from == 'train':
            n = self.data.size
            index = kernel.random.randint(0, n)
            return self.data.get_single_train(index)
        elif self.sample_from == 'valid':
            n = self.data.valid_size
            index = kernel.random.randint(0, n)
            return self.data.get_single_valid(index)
        elif self.sample_from == 'test':
            n = self.data.test_size
            index = kernel.random.randint(0, n)
            return self.data.get_single_valid(index)


sample = Sample()
