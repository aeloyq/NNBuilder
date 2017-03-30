# -*- coding: utf-8 -*-
"""
Created on  Feb 16 1:28 AM 2017

@author: aeloyq
"""


import numpy as np
import theano
import theano.tensor as T
from layers import hidden_layer,layer_tools

''' setup hidden layer of feedforward network inherited from Hidden_Layer '''


class output_ways:
    def __init__(self):
        self.final = layer_tools.final
        self.all = layer_tools.all
        self.mean_pooling = layer_tools.mean_pooling

class get(hidden_layer):
    def __init__(self, in_dim, unit_dim, h_0_init=False, activation=T.tanh, **kwargs):
        hidden_layer.__init__(self, in_dim, unit_dim,activation,**kwargs)
        self.h_0_init=h_0_init
        self.param_init_function = {'wt': self.param_init_functions.uniform, 'bi': self.param_init_functions.zeros,
                                    'u': self.param_init_functions.orthogonal,'h_0':self.param_init_functions.zeros}
        self.u = 'u'
        self.params = [self.wt, self.u, self.bi]
        if self.h_0_init:
            self.h_0= 'h_0'
            self.params.append(self.h_0)
        self.output_ways=output_ways()
        self.output_way =self.output_ways.final
        self.hidden_unit_dropout=True
        self.output_dropout = False

    def init_layer_params(self):
        wt_values = self.param_init_function['wt'](self.in_dim,self.unit_dim)
        bi_values = self.param_init_function['bi'](self.unit_dim)
        u_values=self.param_init_function['u'](self.unit_dim,self.unit_dim)
        self.wt = theano.shared(value=wt_values, name='Wt'+'_'+self.name, borrow=True)
        self.bi = theano.shared(value=bi_values, name='Bi'+'_'+self.name, borrow=True)
        self.u = theano.shared(value=u_values, name='U' + '_' + self.name, borrow=True)
        self.params = [self.wt, self.u, self.bi]
        if self.h_0_init:
            h_0_values=self.param_init_function['h_0'](self.unit_dim)
            self.h_0=theano.shared(value=h_0_values, name='h_0' + '_' + self.name, borrow=True)
            self.params = [self.wt, self.u, self.h_0, self.bi]

    def set_x_mask(self,tvar):
        self.x_mask=tvar

    def get_state_before(self):
        self.state_before=T.dot(self.input, self.wt)+ self.bi

    def get_n_samples(self):
        if self.input.ndim == 3:
            n_samples = self.input.shape[1]
        else:
            n_samples = 1
        self.n_samples=n_samples

    def get_output(self):
        self.get_state_before()
        self.get_n_samples()
        h_0=T.alloc(np.asarray(0.).astype(theano.config.floatX),
                                    self.n_samples, self.unit_dim)
        if self.h_0_init:h_0=T.reshape(T.tile(self.h_0,self.n_samples),[self.n_samples,self.unit_dim])
        lin_out,scan_update=theano.scan(self.step, sequences=[self.state_before,self.x_mask],
                                        outputs_info=[h_0], name=self.name + '_Scan', n_steps=self.state_before.shape[0])
        self.output = self.output_way(lin_out,self.x_mask)
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)
    def step(self, x_, m_, h_):
        h= T.dot(h_, self.u) + x_
        if self.activation is not None:
            h = self.activation(h)
        else:
            h = h
        h= m_[:, None] * h + (1. - m_)[:, None] * h
        if self.hidden_unit_dropout:
            if self.ops is not None:
                h=self.ops(h)
        return h

