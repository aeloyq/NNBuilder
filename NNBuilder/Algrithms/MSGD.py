# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:05 PM 2017

@author: aeloyq
"""

import theano
import theano.tensor as T

class algrithm:
    def __init__(self,configuration,wt_packs):
        self.params=[theta for param in wt_packs for theta in param]
        self.configuration=configuration
        self.gparams = []
        for param in self.params:
            if param.ndim==2:
                self.gparams.append(T.matrix())
            if param.ndim==1:
                self.gparams.append(T.vector())
    def set_params(self,wt_packs):
        self.params = [theta for param in wt_packs for theta in param]
    def get_updates(self):
        updates=[(param, param - self.configuration['learning_rate'] * gparam)
               for param, gparam in zip(self.params, self.gparams)]
        return updates
    def get_update_func(self):
        fn=theano.function(inputs=self.gparams,updates=self.get_updates())
        return fn
    def repeat(self,argv):
        pass
    def save_p_grad(self,p_grad):
        pass