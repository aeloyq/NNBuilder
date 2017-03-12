# -*- coding: utf-8 -*-
"""
Created on  Feb 24 4:05 AM 2017

@author: aeloyq
"""

import theano
import theano.tensor as T
import numpy as np
import sgd

base=sgd.algrithm
class algrithm(base):
    def __init__(self):
        base.__init__(self)
    def init(self,wt_packs,cost):
        base.init(self, wt_packs, cost)
        self.pro_rms_g2 = [theano.shared(param.get_value() * self.numpy_floatX(0.),
                                         name='rmsprop_pro_rms_g2_%s'%param.name,borrow=True)
                           for param in self.params]
    def get_updates(self):
        self.gparams = T.grad(self.cost, self.params)
        self.rou=0.5
        self.epsilon = 1
        learning_rate = theano.shared(self.numpy_floatX(self.learning_rate), name='Learning_Rate')
        self.learning_rate = learning_rate
        self.current_g2 = [(self.rou*p_g2+(1-self.rou)* (gparam**2))
                           for gparam, p_g2 in zip(self.gparams, self.pro_rms_g2)]
        self.current_update = [((learning_rate/(T.sqrt(curnt_g2+self.epsilon)))*gparam)
                           for gparam, curnt_g2 in zip(self.gparams, self.current_g2)]
        self.train_updates=[(param, param - curnt_updt)
               for param, curnt_updt in zip(self.params, self.current_update)]
        self.updates=[(p_g2, curnt_g2)
                      for p_g2, curnt_g2 in zip(self.pro_rms_g2, self.current_g2)]
        return self.train_updates+self.updates
    #TODO: there was a quick algrithm similar to this one

config=algrithm()