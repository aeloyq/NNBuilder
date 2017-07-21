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
        self.t=1

    def init(self, wrt, cost):
        base.init(self, wrt, cost)
        alpha = theano.shared(self.numpy_floatX(self.alpha)
                                                    , name='alpha')
        self.beta_1_= theano.shared(self.numpy_floatX(self.beta_1)
                                    , name='beta_1')
        self.beta_2_ = theano.shared(self.numpy_floatX(self.beta_2)
                                     , name='beta_2')
        self.epsilon_ = theano.shared(self.numpy_floatX(self.epsilon)
                                      , name='epsilon')
        self.t_ = theano.shared(self.numpy_floatX(self.t)
                                , name='t')
        self.alpha=alpha * T.sqrt(1. - self.beta_2_ ** self.t_) / (1. - self.beta_1_ ** self.t_)
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
        self.iter_dict_(lambda x, y: self.beta_1_ * y + (1 - self.beta_1_) * x, self.gparams,
                        self.m, self.cm)
        self.iter_dict_(lambda x, y: self.beta_2_ * y + (1 - self.beta_2_) * (x ** 2), self.gparams,
                        self.v, self.cv)
        self.iter_dict(lambda x: x/(1 - self.beta_1_), self.cm,
                       self.m_)
        self.iter_dict(lambda x: x/(1 - self.beta_2_), self.cv,
                       self.v_)
        self.iter_dict_(lambda x,y: (self.alpha*x)/(T.sqrt(y) + self.epsilon_), self.m_,
                        self.v_, self.updates2output)
        self.iter_updates()
        for name, pnew in self.cm.items():
            self.updates_m[self.m[name]] = pnew
        for name, pnew in self.cv.items():
            self.updates_v[self.v[name]] = pnew
        self.updates.update(self.updates_m)
        self.updates.update(self.updates_v)
        self.updates.update({self.t_: self.t_ + 1})
        return self.updates
    def save_(self,dict):
        dict['optimizer']={'m':{},'v':{}}
        for m in self.m.values():
            dict['optimizer']['m'][m.name]=m.get_value()
        for v in self.v.values():
            dict['optimizer']['v'][v.name]=v.get_value()
        dict['optimizer']['t']=self.t_.get_value()
    def load_(self,dict):
        for m in self.m.values():
            m.set_value(dict['optimizer']['m'][m.name])
        for v in self.v.values():
            v.set_value(dict['optimizer']['v'][v.name])
        self.t_.set_value(dict['optimizer']['t'])

config = algrithm()

