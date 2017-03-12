# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 14:30:08 2017

@author: aeloyq
"""

import numpy as np
import theano.tensor as T
from layers import output_layer

''' setup logistic output layer inherited from base output layer '''

class get_new(output_layer):
    def __init__(self,in_dim, unit_dim,Activation=T.nnet.sigmoid):
       output_layer.__init__(self,in_dim,unit_dim, Activation)
       self.cost_function=self.cost_functions.cross_entropy