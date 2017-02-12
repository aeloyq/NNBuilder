# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:55:42 2016

@author: aeloyq
"""

import numpy as np
import theano.tensor as T
from Layers import Hidden_Layer

''' setup hidden layer of feedforward network inherited from Hidden_Layer '''

class Hidden_Layer_FeedForward(Hidden_Layer):
    def __init__(self,Rng,N_in,N_hl,Wt=None,Bi=None,Wt_init='uniform',Bi_init='zeros',Activation=T.nnet.sigmoid):
        Hidden_Layer.__init__(self,Rng,N_in,N_hl,Wt,Bi,Wt_init,Bi_init,Activation)    