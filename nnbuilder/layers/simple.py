# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:55:42 2016

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from basic import baselayer,layer,hidden_layer,output_layer
from utils import *
from roles import *
from ops import *



class hiddenlayer(hidden_layer):
    ''' 
    setup hidden layer of feedforward network inherited from Hidden_Layer
    '''
    def __init__(self, in_dim, unit_dim, activation=T.tanh):
        hidden_layer.__init__(self,in_dim, unit_dim, activation)

class embedding(layer):
    def __init__(self,in_dim,emb_dim,**kwargs):
        layer.__init__(self,**kwargs)
        self.in_dim = in_dim
        self.emb_dim = emb_dim
    def init_params(self):
        wemb_values=self.param_init_function['wemb'](self.in_dim, self.emb_dim)
        self.wemb=theano.shared(value=wemb_values,name='Wemb'+'_'+self.name,borrow=True)
        self.params = [self.wemb]
    def get_output(self):
        n_timesteps = self.input.shape[0]
        n_samples =   self.input.shape[1]
        self.output=T.reshape(self.wemb[self.input.flatten()] ,[n_timesteps,
                                                    n_samples,self.emb_dim])
        if self.ops is not None:
            self.output = self.ops(self.output)