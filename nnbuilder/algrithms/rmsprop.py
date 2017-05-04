# -*- coding: utf-8 -*-
"""
Created on  Feb 24 4:05 AM 2017

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
        self.rou=0.5
        self.epsilon = 1
    def init(self,wt_packs,cost):
        base.init(self, wt_packs, cost)
        self.rou = theano.shared(self.numpy_floatX(self.rou)
                                                    , name='rou')
        self.epsilon = theano.shared(self.numpy_floatX(self.epsilon)
                                 , name='epsilon')
        self.learning_rate = theano.shared(self.numpy_floatX(self.learning_rate), name='Learning_Rate')
        self.pEg2 =OrderedDict()
        self.cEg2 = OrderedDict()
        self.updates_Eg2=OrderedDict()
        self.iter_dict(lambda x: theano.shared(x.get_value() * self.numpy_floatX(0.),
                                               name='rmsprop_pro_rms_g2_%s' % x.name, borrow=True), self.params, self.pEg2)
    def get_updates(self):
        self.get_grad()
        self.iter_dict_(lambda x, y: self.rou * y + (1 - self.rou) * (x ** 2), self.gparams, self.pEg2, self.cEg2)
        self.iter_dict_(lambda x, y, z: (self.learning_rate/T.sqrt(y+self.epsilon))*x, self.gparams,
                         self.cEg2, self.updates2output)
        self.iter_updates()
        for name,pnew in self.cEg2.items():
            self.updates_Eg2[self.pEg2[name]]=pnew
        self.updates.update(self.updates_Eg2)
        return self.updates
    #TODO: there was a quick algrithm similar to this one

config=algrithm()