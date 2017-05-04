# -*- coding: utf-8 -*-
"""
Created on  Feb 14 2:02 AM 2017

@author: aeloyq
"""

import theano
import theano.tensor as T
import numpy as np
import sgd
from collections import OrderedDict

base=sgd.algrithm
class algrithm(base):
    def __init__(self):
        base.__init__(self)
        self.learning_rate=0.01
        self.momentum_factor=0.9
    def init(self,wt_packs,cost):
        base.init(self, wt_packs, cost)
        self.momentum_factor_shared=theano.shared(self.numpy_floatX(self.momentum_factor)
                                           ,name='momentum_factor')
        self.pu=OrderedDict()
        self.iter_dict(lambda x: theano.shared(x.get_value() * self.numpy_floatX(0.),
                                               name='rmsprop_pro_rms_g2_%s' % x.name, borrow=True), self.params,
                       self.pu)
        self.updates_pu=OrderedDict()
    def get_updates(self):
        self.get_grad()
        self.iter_dict_(lambda x, y: self.momentum_factor_shared * y + self.learning_rate * x, self.gparams,
                        self.pu, self.updates2output)
        self.iter_updates()
        for name, pnew in self.updates2output.items():
            self.updates_pu[self.updates2output[name]] = pnew
        self.updates.update(self.updates_pu)
        return self.updates
    #TODO: the grad updated to two differen shared variable is that incline to time-consuming?need for test

config=algrithm()