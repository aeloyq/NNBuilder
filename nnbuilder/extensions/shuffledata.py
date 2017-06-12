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
        self.shuffle_window=None
    def before_train(self):
        if self.kwargs['idx']==0:
            self.kwargs['minibatch']=self.kwargs['get_minibatches_idx'](self.kwargs['data_stream'],True,self.shuffle_window)

    def after_epoch(self):
        self.kwargs['minibatch'] = self.kwargs['get_minibatches_idx'](self.kwargs['data_stream'], True,
                                                                      self.shuffle_window)

config=ex({})