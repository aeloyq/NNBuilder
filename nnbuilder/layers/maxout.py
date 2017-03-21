# -*- coding: utf-8 -*-
"""
Created on  Feb 16 1:27 AM 2017

@author: aeloyq
"""


import numpy as np
import theano.tensor as T
from layers import hidden_layer

''' setup hidden layer of feedforward network inherited from Hidden_Layer '''

class get(hidden_layer):
    def __init__(self, Rng, in_dim, N_hl, Name='undefined', Wt=None, Bi=None, Wt_init='uniform', Bi_init='zeros', Activation=T.argmax):
        hidden_layer.__init__(self, Rng, in_dim, N_hl, Name, Wt, Bi, Wt_init, Bi_init, Activation)