# -*- coding: utf-8 -*-
"""
Created on  Feb 14 2:02 AM 2017

@author: aeloyq
"""

import theano
import theano.tensor as T
import numpy as np
import SGD

base=SGD.algrithm
class algrithm(base):
    def __init__(self,configuration,wt_packs,cost):
        base.__init__(self,configuration,wt_packs,cost)
        self.pro_update = [theano.shared(param.get_value() * self.numpy_floatX(0.),
                              name='momentum_pro_update_%s'%param.name,borrow=True)
                                            for param in self.params]
    def get_updates(self):
        self.gparams = T.grad(self.cost, self.params)
        self.momentum_factor=theano.shared(self.numpy_floatX(self.configuration['momentum_factor'])
                                           ,name='momentum_factor')
        learning_rate = theano.shared(self.numpy_floatX(self.configuration['learning_rate']), name='Learning_Rate')
        self.learning_rate = learning_rate
        self.train_updates_current_delta_x = [self.momentum_factor*p_grad+learning_rate * gparam for
                            gparam, p_grad in zip(self.gparams, self.pro_update)]
        self.train_updates=[(param, param - updt_curnt)
               for param, updt_curnt in zip(self.params, self.train_updates_current_delta_x)]
        self.updates=[(pro_updt, updt_curnt)
               for pro_updt, updt_curnt in zip(self.pro_update, self.train_updates_current_delta_x)]
        return self.train_updates+self.updates
    #TODO: the grad updated to two differen shared variable is that cost more time?need to be tested