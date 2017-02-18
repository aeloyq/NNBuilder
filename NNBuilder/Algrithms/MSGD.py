# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:05 PM 2017

@author: aeloyq
"""

import theano
import theano.tensor as T

class algrithm:
    def __init__(self,configuration,wt_packs,cost):
        self.params=[theta for param in wt_packs for theta in param]
        self.configuration=configuration
        self.cost=cost
    def set_params(self,wt_packs):
        self.params = [theta for param in wt_packs for theta in param]
    def get_updates(self):
        self.gparams = T.grad(self.cost, self.params)
        self.updates2output=[self.configuration['learning_rate'] * gparam for
                            gparam in self.gparams]
        self.updates=[(param, param - lrgp)
               for param, lrgp in zip(self.params, self.updates2output)]
        return self.updates