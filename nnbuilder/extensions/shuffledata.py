# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import extension
import timeit

base=extension.extension
class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.shuffle_window=-1
    def init(self):
        base.init()
    def after_epoch(self):
        pass

