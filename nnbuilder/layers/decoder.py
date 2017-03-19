# -*- coding: utf-8 -*-
"""
Created on  三月 19 2:29 2017

@author: aeloyq
"""
import theano
import theano.tensor as T
import numpy as np


import layers.baselayer as base

class get_new_lstm_attention(base):
    def __init__(self,in_dim,unit_dim,emb_dim,h_0_init=False,c_0_init=False,activation=T.tanh):
        base.__init__(self)
        self.in_dim = in_dim
        self.unit_dim = unit_dim
        self.emb_idm=emb_dim
        self.h_0_init = h_0_init
        self.c_0_init = c_0_init
        self.param_init_function = {'wt': self.param_init_functions.uniform,
                                    'bi': self.param_init_functions.zeros,
                                    'u':self.param_init_functions.orthogonal,
                                    'h_0':self.param_init_functions.zeros,
                                    'c_0':self.param_init_functions.zeros}
        self.wt='wt'
        self.u='u'
        self.h_0='h_0'
        self.c_0='c_0'
        self.bi='bi'
        self.params=[self.wt,self.u]
        if self.h_0_init: self.params += self.h_0
        if self.c_0_init: self.params += self.c_0
        self.params+=self.bi
    def set_input(self,X):
        self.input=T.transpose(X,(1,0,2))
    def init_layer_params(self):
        wt_values = self.param_init_function['wt'](self.in_dim,self.unit_dim)
        bi_values = self.param_init_function['bi'](self.unit_dim)
        u_values=self.param_init_function['u'](self.unit_dim,self.unit_dim*4)
        self.wt = theano.shared(value=wt_values, name='Wt'+'_'+self.name, borrow=True)
        self.bi = theano.shared(value=bi_values, name='Bi'+'_'+self.name, borrow=True)
        self.u = theano.shared(value=u_values, name='U' + '_' + self.name, borrow=True)
        if self.h_0_init:
            h_0_values = self.param_init_function['h_0'](self.unit_dim)
            self.h_0 = theano.shared(value=h_0_values, name='H_0' + '_' + self.name, borrow=True)
        if self.c_0_init:
            c_0_values = self.param_init_function['c_0'](self.unit_dim)
            self.c_0 = theano.shared(value=c_0_values, name='C_0' + '_' + self.name, borrow=True)
        self.params = [self.wt, self.u]
        if self.h_0_init: self.params += self.h_0
        if self.c_0_init: self.params += self.c_0
        self.params += self.bi
    def get_output(self):
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step(m_, x_, h_, c_):
            preact = T.dot(h_, self.u)
            preact += x_

            i = T.nnet.sigmoid(_slice(preact, 0, self.unit_dim))
            f = T.nnet.sigmoid(_slice(preact, 1, self.unit_dim))
            o = T.nnet.sigmoid(_slice(preact, 2, self.unit_dim))
            c = T.tanh(_slice(preact, 3, self.unit_dim))

            c = f * c_ + i * c
            c = m_[:, None] * c + (1. - m_)[:, None] * c_

            if self.cell_unit_dropout:
                if self.ops is not None:
                    c = self.ops(c)

            h = o * T.tanh(c)
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

            if self.hidden_unit_dropout:
                if self.ops is not None:
                    h = self.ops(h)

            return h, c

        if self.input.ndim == 3:
            n_samples = self.input.shape[1]
        else:
            n_samples = 1
        state_below = T.dot(self.input, self.wt)+self.bi
        lin_out, scan_update = theano.scan(_step, sequences=[self.mask,state_below],
                                           outputs_info=[
                                               T.alloc(np.asarray(0.).astype(theano.config.floatX),
                                                       n_samples, self.unit_dim),T.alloc(np.asarray(0.).astype(theano.config.floatX),
                                                                                         n_samples, self.unit_dim)], name=self.name + '_Scan',
                                           n_steps=self.input.shape[0])
        lin_out=lin_out[0]
        self.output=self.output_way(lin_out,self.mask)
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)