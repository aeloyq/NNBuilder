# -*- coding: utf-8 -*-
"""
Created on  Feb 25 5:02 PM 2017

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from layers import utils,baselayer

class get(baselayer):
    def __init__(self,in_dim,emb_dim,**kwargs):
        baselayer.__init__(self)
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