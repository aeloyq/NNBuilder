# -*- coding: utf-8 -*-
"""
Created on  Feb 25 6:15 PM 2017

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from layers import hidden_layer

''' setup hidden layer of feedforward network inherited from Hidden_Layer '''

class get(hidden_layer):
    def __init__(self, in_dim, unit_dim, activation=T.tanh):
        hidden_layer.__init__(self,in_dim, unit_dim, activation)

class get_deep(hidden_layer):
    def __init__(self, in_dim, unit_dim, activation=T.tanh):
        hidden_layer.__init__(self,in_dim, unit_dim, activation)