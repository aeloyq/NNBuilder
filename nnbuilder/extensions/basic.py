# -*- coding: utf-8 -*-
"""
Created on  Feb 25 1:45 PM 2017

@author: aeloyq
"""
import numpy as np
from nnbuilder.kernel import *
from collections import OrderedDict


class ExtensionBase(object):
    def build(self, MainLoop, model, data, extensions, config, logger, train_history):
        self.MainLoop = MainLoop
        self.model = model
        self.data = data
        self.extensions = extensions
        self.config = config
        self.logger = logger
        self.train_history = train_history
        self.init()

    def init(self):
        pass

    def get_verbose(self):
        return self.config.verbose

    def is_log_detail(self):
        return self.config.is_log_detail()

    def is_log_inline(self):
        return self.config.is_log_inline()

    def is_log_silent(self):
        return self.config.is_log_silent()

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
