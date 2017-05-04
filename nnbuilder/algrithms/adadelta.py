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
        self.rou=0.95
        self.epsilon = 1e-6

    def init(self,wt_packs,cost):
        base.init(self,wt_packs, cost)
        self.rou = theano.shared(self.numpy_floatX(self.rou)
                                                    , name='rou')
        self.epsilon = theano.shared(self.numpy_floatX(self.epsilon)
                                 , name='epsilon')
        self.pEg2 =OrderedDict()
        self.pEdx2=OrderedDict()
        self.cEg2 =OrderedDict()
        self.cEdx2=OrderedDict()
        self.train_updates=OrderedDict()
        self.updates_1=OrderedDict()
        self.updates_2=OrderedDict()
        self.iter_dict(lambda x:theano.shared(x.get_value() * self.numpy_floatX(0.),
                                         name='adadelta_pro_g2_%s' % x.name, borrow=True), self.params, self.pEg2)
        self.iter_dict(lambda x: theano.shared(x.get_value() * self.numpy_floatX(0.),
                                               name='adadelta_pro_delta_x_%s' % x.name, borrow=True), self.params,
                       self.pEdx2)
        self.updates = OrderedDict()

    def get_updates(self):
        self.get_grad()
        self.iter_dict_(lambda x,y: self.rou*y+(1-self.rou)* (x**2), self.gparams, self.pEg2, self.cEg2)
        self.iter_dict__(lambda x, y, z: (T.sqrt(z + self.epsilon) / T.sqrt(y + self.epsilon))*x, self.gparams,
                         self.cEg2, self.pEdx2, self.updates2output)
        self.iter_dict_(lambda x,y : self.rou*y+(1-self.rou)*(x**2), self.updates2output, self.pEdx2, self.cEdx2)
        self.iter_updates()
        for name,pnew in self.cEg2.items():
            self.updates_1[self.pEg2[name]]=pnew
        for name,pnew in self.cEdx2.items():
            self.updates_2[self.pEdx2[name]]=pnew
        self.updates.update(self.updates_1)
        self.updates.update(self.updates_2)
        return self.updates

config=algrithm()

#TODO:change variable names to a unified form