# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 14:30:08 2017

@author: aeloyq
"""

import numpy as np
import theano.tensor as T
from Layers import Output_Layer

''' setup logistic output layer inherited from base output layer '''

class MLP_Logistic(Output_Layer):
    def __init__(self,Rng,N_in,N_out,Raw_Input=None,l0=0.,l1=0.,l2=0.,Wt=None,Bi=None,Wt_init='uniform',Bi_init='zeros',Hidden_Layer_Struct=[],Cost_func='square',Activation=T.nnet.sigmoid):
       Output_Layer.__init__(self,Rng,N_in,N_out,Raw_Input,l0,l1,l2,Wt,Bi,Wt_init,Bi_init,Hidden_Layer_Struct,Cost_func,Activation)