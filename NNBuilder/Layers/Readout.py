# -*- coding: utf-8 -*-
"""
Created on  Feb 25 6:15 PM 2017

@author: aeloyq
"""

import numpy as np
import theano.tensor as T
from Layers import Hidden_Layer

''' setup hidden layer of feedforward network inherited from Hidden_Layer '''

class layer(Hidden_Layer):
    def __init__(self,Rng,N_in,N_hl,Name='undefined',Wt=None,Bi=None,Wt_init='uniform',Bi_init='zeros',Activation=T.tanh):
        Hidden_Layer.__init__(self,Rng,N_in,N_hl,Name,Wt,Bi,Wt_init,Bi_init,Activation)