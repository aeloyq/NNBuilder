# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 14:30:08 2017

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from basic import hidden_layer, utils,baselayer

''' setup softmax output layer inherited from base output layer '''


class get(baselayer):
    def __init__(self,i0,i1,**kwargs):
        baselayer.__init__(self)
        self.i0=i0
        self.i1=i1
    def get_output(self):
        self.output=self.i0.output+self.i1.output
        if self.ops is not None:
            self.output = self.ops(self.output)

