# -*- coding: utf-8 -*-
"""
Created on  Feb 25 1:45 PM 2017

@author: aeloyq
"""

import sys

sys.path.append('..')


class base(object):
    def __init__(self, kwargs):
        self.kwargs = kwargs
        self.set_logattr()

    def init(self):
        self.logger = self.kwargs['logger']

    def before_train(self):
        pass

    def before_init_iter(self):
        pass

    def after_init_iter(self):
        pass

    def before_iteration(self):
        pass

    def after_iteration(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def after_train(self):
        pass

    def save_(self, dict):
        pass

    def load_(self, dict):
        pass

    def set(self, **kwargs):
        for key, val in kwargs.items():
            self.__setattr__(key, val)

    def set_logattr(self):
        self.logattr = []
