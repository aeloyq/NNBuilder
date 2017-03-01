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
    def __init__(self):
        base.__init__(self)
        self.pro_rms_g2 = [theano.shared(param.get_value() * self.numpy_floatX(0.),
                                         name='rmsprop_pro_rms_g2_%s'%param.name,borrow=True)
                           for param in self.params]
        self.pro_rms_delta_x = [theano.shared(param.get_value() * self.numpy_floatX(0.),
                                              name='rmsprop_pro_rms_delta_x_%s' % param.name,borrow=True)
                                for param in self.params]
    def get_updates(self):
        self.gparams = T.grad(self.cost, self.params)
        self.rou=0.95
        self.epsilon = 1e-6
        self.current_g2 = [(self.rou*p_g2+(1-self.rou)* (gparam**2))
                           for gparam, p_g2 in zip(self.gparams, self.pro_rms_g2)]
        self.current_delta_x = [((T.sqrt(prp_dlt_x + self.epsilon) / (T.sqrt(curnt_g2 + self.epsilon))) * gparam)
                                for gparam, curnt_g2,prp_dlt_x in zip(self.gparams, self.current_g2, self.pro_rms_delta_x)]
        self.train_updates=[(param, param - curnt_updt)
                            for param, curnt_updt in zip(self.params, self.current_delta_x)]
        self.updates_1=[(p_g2, curnt_g2)
                      for p_g2, curnt_g2 in zip(self.pro_rms_g2, self.current_g2)]
        self.updates_2 = [(p_g2, self.rou*p_g2+(1-self.rou)*(curnt_dlt_x**2))
                          for p_g2, curnt_dlt_x in zip(self.pro_rms_delta_x, self.current_delta_x)]
        self.updates=self.updates_1+self.updates_2
        return self.train_updates+self.updates

config=algrithm()