# -*- coding: utf-8 -*-
"""
Created on  Feb 14 2:02 AM 2017

@author: aeloyq
"""

import theano
import theano.tensor as T
import MSGD

base=MSGD.algrithm
class algrithm(base):
    def __init__(self,configuration,wt_packs):
        base.__init__(self,configuration,wt_packs)
        self.p_grads=[]
        for param in self.params:
            self.p_grads.append(0)
    def get_updates(self):
        updates=[(param, param - (self.configuration['momentum_factor']*p_grad+self.configuration['learning_rate'] * gparam))
               for param, gparam,p_grad in zip(self.params, self.gparams,self.p_grads)]
        return updates
    def repeat(self,argv):
        self.save_p_grad(argv)
    def save_p_grad(self, p_grads):
        self.p_grads=p_grads