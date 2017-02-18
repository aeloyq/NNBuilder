# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 14:30:08 2017

@author: aeloyq
"""

import numpy as np
import theano.tensor as T
from Layers import Output_Layer

''' setup logistic output layer inherited from base output layer '''

class layer(Output_Layer):
    def __init__(self,Rng,N_in,N_out,Name='undefined',Wt=None,Bi=None,Wt_init='zeros',Bi_init='zeros',Cost_func='cross_entropy',Activation=T.nnet.sigmoid):
       Output_Layer.__init__(self,Rng,N_in,N_out,Name,Wt,Bi,Wt_init,Bi_init,Cost_func,Activation)