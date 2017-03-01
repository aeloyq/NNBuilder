# -*- coding: utf-8 -*-
"""
Created on  Feb 25 5:02 PM 2017

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from Layers import Layer_Tools,baselayer

class layer(baselayer):
    def __init__(self, Rng, N_in, N_dims, Name='undefined', Wemb=None, Wemb_init='randn'):
        baselayer.__init__(self)
        self.Rng=Rng
        self.N_in = N_in
        self.N_dims = N_dims
        self.Wemb=Wemb
        self.Wemb_init=Wemb_init
        self.Inputs=None
        self.wt_bi_inited = False
    def init_wt_bi(self):
        if not self.wt_bi_inited:
            Wemb_values=Layer_Tools.Fully_connected_emb_init(self.Rng,self.N_in,self.N_dims,self.Wemb,self.Wemb_init)
            Wemb=theano.shared(value=Wemb_values,name='Wemb'+'_'+self.Name,borrow=True)
            self.Wemb=Wemb
            self.wt_bi_pack()
            self.wt_bi_inited = True
    def wt_bi_pack(self):
        self.params=[self.Wemb]
    def output_func(self):
        n_timesteps = self.Inputs.shape[0]
        n_samples =   self.Inputs.shape[1]
        self.outputs=self.Wemb[T.cast(self.Inputs.flatten(),'int64')].reshape([n_timesteps,
                                                    n_samples,self.N_dims])
    def set_inputs(self,Inputs_X):
        self.Inputs=Inputs_X
        self.output_func()
    def set_name(self,name):
        self.Name=name