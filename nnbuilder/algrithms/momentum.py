# -*- coding: utf-8 -*-
"""
Created on  Feb 14 2:02 AM 2017

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
        self.learning_rate=0.01
        self.momentum_factor=0.9
    def init(self,wt_packs,cost):
        base.init(self, wt_packs, cost)
        self.pro_update = [theano.shared(param.get_value() * self.numpy_floatX(0.),
                              name='momentum_pro_update_%s'%param.name,borrow=True)
                                            for param in self.params]
    def get_updates(self):
        self.gparams = T.grad(self.cost, self.params)
        self.momentum_factor_shared=theano.shared(self.numpy_floatX(self.momentum_factor)
                                           ,name='momentum_factor')
        self.learning_rate_shared = theano.shared(self.numpy_floatX(self.learning_rate), name='Learning_Rate')
        self.train_updates_current_delta_x = [self.momentum_factor_shared*p_grad+self.learning_rate_shared * gparam for
                            gparam, p_grad in zip(self.gparams, self.pro_update)]
        if self.if_clip: self.train_updates_current_delta_x = [self.grad_clip(update2output) for update2output in self.train_updates_current_delta_x]
        self.train_updates=[(param, param - updt_curnt)
               for param, updt_curnt in zip(self.params, self.train_updates_current_delta_x)]
        self.updates=[(pro_updt, updt_curnt)
               for pro_updt, updt_curnt in zip(self.pro_update, self.train_updates_current_delta_x)]
        return self.train_updates+self.updates
    #TODO: the grad updated to two differen shared variable is that incline to time-consuming?need for test

config=algrithm()