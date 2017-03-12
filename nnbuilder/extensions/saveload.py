# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:45:34 2016

@author: aeloyq
"""
import extension
import time
import numpy as np

base=extension.extension
class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.save_freq=10000
        self.save_len=3
        self.load=True
        self.save=True
    def init(self):
        base.init(self)
    def before_train(self):
        kwargs = self.kwargs
        wt2save=kwargs['model_stream'][7].wt_packs
        loadname=''
        np.load('./Saves/%s/%s'%(kwargs['conf'].name,loadname),wt2save)
    def after_train(self):
        kwargs = self.kwargs
        wt2save=kwargs['model_stream'][7].wt_packs
        np.save('./Saves/%s/%s'%(kwargs['conf'].name,time.localtime()),wt2save)
config=ex({})

