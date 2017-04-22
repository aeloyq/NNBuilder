# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 14:30:08 2017

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from layers import output_layer, utils,baselayer,costfunctions




class get(baselayer):
    ''' setup direct output layer inherited from base output layer '''
    def __init__(self,**kwargs):
        baselayer.__init__(self)
        self.cost_fn = utils.square_cost
        self.cost = None
        self.predict = None
        self.error = None
        if 'cost_fn' in kwargs:
            self.cost_fn = kwargs['cost_fn']
    def get_predict(self):
        self.predict=T.round(self.output)
    def get_cost(self,Y):
        self.cost=self.cost_fn(Y,self.output)
    def get_error(self,Y):
        self.error=T.mean(T.neq(Y,self.predict))

