# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:05 PM 2017

@author: aeloyq
"""

import theano
import numpy as np
import theano.tensor as T

class algrithm:
    def __init__(self):
        self.params=[]
        self.configuration={}
        self.cost=None
        self.learning_rate=0.01
        self.if_clip=False
        self.grad_clip_norm=1.
    def init(self,wt_packs,cost):
        self.params = [theta for param in wt_packs for theta in param]
        self.cost = cost
    def set_params(self,wt_packs):
        self.params = [theta for param in wt_packs for theta in param]
    def get_updates(self):
        self.gparams = T.grad(self.cost, self.params)
        learning_rate=theano.shared(self.numpy_floatX(self.learning_rate),name='Learning_Rate',borrow=True)
        self.updates2output=[learning_rate * gparam for
                            gparam in self.gparams]
        self.updates2output = [0.01 * gparam for
                               gparam in self.gparams]
        if self.if_clip:self.updates2output=[self.grad_clip(update2output) for update2output in self.updates2output]
        self.updates=[(param, param - lrgp)
               for param, lrgp in zip(self.params, self.updates2output)]
        return self.updates
    def grad_clip(self,grad):
        return T.clip(grad,-self.grad_clip_norm,self.grad_clip_norm)
    def numpy_floatX(self,data):
        return np.asarray(data, dtype=theano.config.floatX)
config=algrithm()