# -*- coding: utf-8 -*-
"""
Created on  Feb 25 1:45 PM 2017

@author: aeloyq
"""

import sys
sys.path.append('..')

class extension:
    def __init__(self,kwargs):
        self.kwargs=kwargs
    def init(self):
        self.logger=self.kwargs['logger']
    def before_train(self):
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
    def save_(self,dict):
        pass
    def load_(self,dict):
        pass