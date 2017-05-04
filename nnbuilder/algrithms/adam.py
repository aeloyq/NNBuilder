# -*- coding: utf-8 -*-
"""
Created on  四月 25 18:10 2017

@author: aeloyq
"""
# -*- coding: utf-8 -*-

import theano
import theano.tensor as T
import numpy as np
import sgd
from collections import OrderedDict

base=sgd.algrithm
class algrithm(base):
    def __init__(self):
        base.__init__(self)
        self.alpha=0.001
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8

    def init(self, wt_packs, cost):
        base.init(self, wt_packs, cost)
        self.alpha = theano.shared(self.numpy_floatX(self.alpha)
                                                    , name='alpha')
        self.beta_1= theano.shared(self.numpy_floatX(self.beta_1)
                                                    , name='beta_1')
        self.beta_2 = theano.shared(self.numpy_floatX(self.beta_2)
                                                    , name='beta_2')
        self.epsilon = theano.shared(self.numpy_floatX(self.epsilon)
                                 , name='epsilon')
        self.m= OrderedDict()
        self.v = OrderedDict()
        self.cm= OrderedDict()
        self.cv = OrderedDict()
        self.updates_m = OrderedDict()
        self.updates_v = OrderedDict()
        self.m_ = OrderedDict()
        self.v_ = OrderedDict()
        self.updates2output=OrderedDict()
        self.iter_dict(lambda x: theano.shared(x.get_value() * self.numpy_floatX(0.),
                                               name='adam_m_%s' % x.name, borrow=True), self.params,
                       self.m)
        self.iter_dict(lambda x: theano.shared(x.get_value() * self.numpy_floatX(0.),
                                               name='adam_v_%s' % x.name, borrow=True),
                       self.params,
                       self.v)
        self.updates = OrderedDict()

    def get_updates(self):
        self.get_grad()
        self.iter_dict_(lambda x, y: self.beta_1 * y + (1 - self.beta_1) * x, self.gparams,
                        self.m,self.cm)
        self.iter_dict_(lambda x, y: self.beta_2 * y + (1 - self.beta_2) * (x**2), self.gparams,
                       self.v,self.cv)
        self.iter_dict(lambda x: x/(1 - self.beta_1), self.cm,
                        self.m_)
        self.iter_dict(lambda x: x/(1 - self.beta_2), self.cv,
                        self.v_)
        self.iter_dict_(lambda x,y: (self.alpha*x)/(T.sqrt(y)+self.epsilon), self.m_,
                       self.v_,self.updates2output)
        self.iter_updates()
        for name, pnew in self.cm.items():
            self.updates_m[self.m[name]] = pnew
        for name, pnew in self.cv.items():
            self.updates_v[self.v[name]] = pnew
        self.updates.update(self.updates_m)
        self.updates.update(self.updates_v)
        return self.updates

config = algrithm()

#TODO:change variable names to a unified form