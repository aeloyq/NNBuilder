# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:05 PM 2017

@author: aeloyq
"""

import theano
import numpy as np
import theano.tensor as T
from collections import OrderedDict

class algrithm:
    def __init__(self):
        self.params=None
        self.configuration={}
        self.cost=None
        self.learning_rate=0.01
        self.if_clip=False
        self.grad_clip_norm=1.
    def init(self, wrt, cost):
        self.params = wrt
        self.cost = cost
        self.learning_rate = theano.shared(self.numpy_floatX(self.learning_rate), name='Learning_Rate', borrow=True)
        self.gparams=OrderedDict()
        self.updates2output = OrderedDict()
        self.updates = OrderedDict()
    def get_grad(self):
        self.iter_dict(lambda x:T.grad(self.cost,x),self.params,self.gparams)
        if self.if_clip:
            self.iter_dict(lambda x:self.grad_clip(x),self.gparams,self.gparams)
    def get_updates(self):
        self.get_grad()
        self.iter_dict(lambda x: self.learning_rate*x, self.gparams, self.updates2output)
        self.iter_updates()
        return self.updates
    def grad_clip(self,grad):
        return T.clip(grad,-self.grad_clip_norm,self.grad_clip_norm)
    def numpy_floatX(self,data):
        return np.asarray(data, dtype=theano.config.floatX)
    def iter_dict(self,fn,dict1,dict2):
        for name, elem in dict1.items():
            dict2[name] = fn(elem)
    def iter_dict_(self,fn,dict1,dict2,dict3):
        for name, elem1 in dict1.items():
            elem2=dict2[name]
            dict3[name] = fn(elem1,elem2)
    def iter_dict__(self,fn,dict1,dict2,dict3,dict4):
        for name, elem1  in dict1.items():
            elem2=dict2[name]
            elem3=dict3[name]
            dict4[name] = fn(elem1,elem2,elem3)
    def iter_updates(self):
        for name,delta in self.updates2output.items():
            self.updates[self.params[name]]=self.params[name]-delta
    def save_(self,dict):
        pass
    def load_(self,dict):
        pass
config=algrithm()
base=config