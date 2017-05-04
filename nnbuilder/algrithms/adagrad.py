# -*- coding: utf-8 -*-
"""
Created on  Feb 14 4:37 PM 2017

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
        self.epsilon = 1e-8
    def init(self,wt_packs,cost):
        base.init(self, wt_packs, cost)
        self.epsilon = theano.shared(self.numpy_floatX(self.epsilon)
                                 , name='epsilon')
        self.pu = [theano.shared(param.get_value() * self.numpy_floatX(0.),
                                 name='adagrad_all_pro_update_%s' % param.name, borrow=True)
                   for param in self.params]

        self.iter_dict(lambda x: theano.shared(x.get_value() * self.numpy_floatX(0.),
                                               name='rmsprop_pro_rms_g2_%s' % x.name, borrow=True), self.params,
                       self.pu)
        self.updates_pu=OrderedDict()
    def get_updates(self):
        self.get_grad()
        self.iter_dict_(lambda x, y: (self.learning_rate / (T.sqrt(y+x**2)+self.epsilon))*x, self.gparams,
                        self.pu, self.updates2output)
        self.iter_updates()
        for name, pnew in self.updates2output.items():
            self.updates_pu[self.updates2output[name]] = pnew
        self.updates.update(self.updates_pu)
        return self.updates

config=algrithm()