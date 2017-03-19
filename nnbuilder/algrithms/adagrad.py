# -*- coding: utf-8 -*-
"""
Created on  Feb 14 4:37 PM 2017

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
    def init(self,wt_packs,cost):
        base.init(self, wt_packs, cost)
        self.pro_update = [theano.shared(param.get_value() * self.numpy_floatX(0.),
                                         name='adagrad_all_pro_update_%s' % param.name,borrow=True)
                           for param in self.params]
    def get_updates(self):
        self.gparams = T.grad(self.cost, self.params)
        self.epsilon = 1e-8
        self.learning_rate_shared = theano.shared(self.numpy_floatX(self.learning_rate), name='Learning_Rate')
        self.train_updates_current_delta_x = [
            (self.learning_rate_shared / (T.sqrt(p_grad+gparam**2)+self.epsilon))*gparam for
            gparam, p_grad in zip(self.gparams, self.pro_update)]
        if self.if_clip: self.train_updates_current_delta_x = [self.grad_clip(update2output) for update2output in
                                                               self.train_updates_current_delta_x]
        self.train_updates = [(param, param - updt_curnt)
                              for param, updt_curnt in zip(self.params, self.train_updates_current_delta_x)]

        self.updates = [(pro_updt, pro_updt+g**2)
                        for pro_updt, g in zip(self.pro_update, self.gparams)]
        return self.train_updates + self.updates

config=algrithm()