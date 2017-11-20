# -*- coding: utf-8 -*-
"""
Created on  Feb 25 4:11 PM 2017

@author: aeloyq
"""
import numpy as np
from saveload import saveload
from basic import ExtensionBase
from collections import OrderedDict


class EarlyStop(ExtensionBase):
    def __init__(self):
        ExtensionBase.__init__(self)
        self.freq = None
        self.patience = 10
        self.epoch = True
        self.loss = False
        self.save = False
        self.save_best = True

    def init(self):
        self.valid_losses = []
        self.valid_scores = OrderedDict()
        for m in self.model.metrics:
            self.valid_scores[m.name] = []
        self.best_iter = -1
        self.best_epoch = -1
        self.best_valid_loss = 1e8
        self.best_valid_score = None
        self.bad_count = 0
        if self.epoch:
            if self.freq is None:
                self.freq = 1
        else:
            if self.freq is None:
                self.freq = self.data.size // 5

    def after_iteration(self):
        if not self.epoch:
            if (self.train_history['iter']+1) % self.freq == 0:
                self.valid()

    def after_epoch(self):
        if self.epoch:
            if (self.train_history['n_epoch']) % self.freq == 0:
                self.valid()

    def valid(self):
        valid_result = self.MainLoop.valid(self.model, self.data)
        for m in self.model.metrics:
            self.valid_scores[m.name].append(valid_result[m.name])
        self.valid_losses.append(valid_result['loss'])

        def log(valid_result):
            valid_log = "Valid at Epoch:%d  Iter:%d  Loss:%.4f" % (
                self.train_history['n_epoch'], self.train_history['n_iter'], valid_result['loss'])
            for m in self.model.metrics:
                valid_log += '  '
                valid_log += m.name + ':' + '%.4f' % valid_result[m.name]
            self.logger(valid_log, 1, 0)

        def better_countered(valid_loss, valid_score, valid_result):
            self.bad_count = 0
            self.best_valid_loss = valid_loss
            self.best_valid_score = valid_score
            self.best_iter = self.train_history['n_iter']
            self.best_epoch = self.train_history['n_epoch']
            if self.is_log_detail():
                self.logger("Better Model Detected", 1, 0)
            log(valid_result)
            if self.is_log_detail():
                self.logger('-' * 70, 1)
                self.logger('', 1)
            if self.save_best:
                saveload.mainloop_save(self.model, 'valid/best/', str(self.train_history['n_iter']) + '.npz')

        def bad_countered(valid_result):
            self.bad_count += 1
            if self.is_log_detail():
                self.logger("Patience Remain:%d   Bad Count:%d" % (self.patience - self.bad_count, self.bad_count), 1,
                            0)
            log(valid_result)
            if self.is_log_detail():
                self.logger('-' * 70, 1)
                self.logger('', 1)

        if self.is_log_detail():
            self.logger('', 1)
            self.logger('-' * 70, 1)
        valid_score = valid_result[valid_result.keys()[0]]
        valid_loss = valid_result['loss']
        if not self.loss:
            if self.best_valid_score is None:
                better_countered(valid_loss, valid_score, valid_result)
            else:
                if self.model.metrics[0].down_direction:
                    if valid_score <= self.best_valid_score:
                        better_countered(valid_loss, valid_score, valid_result)
                    else:
                        bad_countered(valid_result)
                else:
                    if valid_score >= self.best_valid_score:
                        better_countered(valid_loss, valid_score, valid_result)
                    else:
                        bad_countered(valid_result)
        else:
            if valid_loss <= self.best_valid_loss:
                better_countered(valid_loss, valid_score, valid_result)
            else:
                bad_countered(valid_result)
        if self.patience == self.bad_count:
            if self.is_log_detail():
                self.logger("")
                self.logger("NO Trainning Patience      Early Stop Soon", 1)
            self.logger("Best Valid Model at Epoch:%d at Iter:%d  Best Loss:%.4f  Best %s:%.4f" % (
                self.best_epoch, self.best_iter, self.best_valid_loss, self.model.metrics[0].name,
                self.best_valid_score), 1)

            self.logger("")
            self.train_history['stop'] = True
        if self.save:
            saveload.mainloop_save(self.model, 'valid/', str(self.train_history['n_iter']) + '.npz', max=self.patience,
                                   overwrite=False)

    def save_(self, dict):
        dict = {'patience': self.patience, 'best_iter': self.best_iter,
                'best_valid_score': self.best_valid_score, 'valid_losses': self.valid_losses,
                'valid_scores': self.valid_scores}
        return dict

    def load_(self, dict):
        try:
            self.patience = dict['patience']
            self.best_iter = dict['best_iter']
            self.best_valid_score = dict['best_valid_score']
            self.valid_losses = dict['valid_losses']
            self.valid_scores = dict['valid_scores']
        except:
            self.logger("Didn't use earlystop in latest training Abort earlystop loading", 2)


earlystop = EarlyStop()
