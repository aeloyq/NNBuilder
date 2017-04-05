# -*- coding: utf-8 -*-
"""
Created on  Feb 16 1:28 AM 2017

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from layers import layer_tools
import recurrent

base = recurrent.get

''' setup hidden layer of feedforward network inherited from Hidden_Layer '''


class get(base):
    def __init__(self, in_dim, unit_dim, h_0_init=False, activation=None, **kwargs):
        base.__init__(self, in_dim, unit_dim, h_0_init, activation, **kwargs)
        self.ug = 'ug'
        self.wg = 'wg'
        self.big = 'big'
        self.masked = True
        self.param_init_function['ug'] = self.param_init_functions.orthogonal
        self.param_init_function['wg'] = self.param_init_functions.uniform
        self.param_init_function['big'] = self.param_init_functions.zeros
        self.params=[self.wt,self.bi,self.wg,self.big,self.u,self.ug]
        self.output_way = self.output_ways.all

    def init_layer_params(self):
        wt_values = self.param_init_function['wt'](self.in_dim, self.unit_dim)
        wg_values = self.param_init_function['wg'](self.in_dim, self.unit_dim * 2)
        bi_values = self.param_init_function['bi'](self.unit_dim)
        big_values = self.param_init_function['big'](self.unit_dim * 2)
        u_values = self.param_init_function['u'](self.unit_dim, self.unit_dim)
        ug_values = self.param_init_function['ug'](self.unit_dim, self.unit_dim * 2)
        self.wt = theano.shared(value=wt_values, name='Wt' + '_' + self.name, borrow=True)
        self.bi = theano.shared(value=bi_values, name='Bi' + '_' + self.name, borrow=True)
        self.u = theano.shared(value=u_values, name='U' + '_' + self.name, borrow=True)
        self.wg = theano.shared(value=wg_values, name='Wg' + '_' + self.name, borrow=True)
        self.big = theano.shared(value=big_values, name='Big' + '_' + self.name, borrow=True)
        self.ug = theano.shared(value=ug_values, name='Ug' + '_' + self.name, borrow=True)
        if self.h_0_init:
            h_0_values = np.zeros(self.unit_dim, 'float32')
            self.h_0 = theano.shared(value=h_0_values, name='H_0' + '_' + self.name, borrow=True)
        self.params = [self.wt, self.bi, self.wg, self.big, self.u, self.ug]
        if self.h_0_init: self.params.append(self.h_0)

    def slice(self, _x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        return _x[:, n * dim:(n + 1) * dim]

    def get_output(self):
        self.get_n_samples()
        h_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), self.n_samples, self.unit_dim)
        if self.h_0_init: h_0 = T.reshape(T.tile(self.h_0, self.n_samples), [self.n_samples, self.unit_dim])
        if self.masked:
            lin_out, scan_update = theano.scan(self.step_mask, sequences=[self.input, self.x_mask],
                                               outputs_info=[h_0], name=self.name + '_Scan',
                                               n_steps=self.input.shape[0])
        else:
            lin_out, scan_update = theano.scan(self.step, sequences=[self.input],
                                               outputs_info=[h_0], name=self.name + '_Scan',
                                               n_steps=self.input.shape[0])
        if self.masked:
            self.output = self.output_way(lin_out, self.x_mask)
        else:
            self.output=lin_out
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)

    def get_state_before(self):
        self.state_before=T.dot(self.input, self.wg)+ self.big

    def step_mask(self, x_, m_, h_):
        preact = T.dot(h_, self.ug) + T.dot(x_, self.wg)+ self.big

        r = T.nnet.sigmoid(self.slice(preact, 0, self.unit_dim))
        z = T.nnet.sigmoid(self.slice(preact, 1, self.unit_dim))

        h_c = T.tanh(T.dot(x_,self.wt)+T.dot(r*h_,self.u)+self.bi)

        h = (1-z)*h_+z*h_c

        if self.masked:
            h = m_[:, None] * h + (1. - m_)[:, None] * h_

        if self.hidden_unit_dropout:
            if self.ops is not None:
                h = self.ops(h)

        return h

    def step(self, x_, h_):
        preact = T.dot(h_, self.ug) + T.dot(x_, self.wg) + self.big

        r = T.nnet.sigmoid(self.slice(preact, 0, self.unit_dim))
        z = T.nnet.sigmoid(self.slice(preact, 1, self.unit_dim))

        h_c = T.tanh(T.dot(x_, self.wt) + T.dot(r * h_, self.u) + self.bi)

        h = (1 - z) * h_ + z * h_c

        if self.hidden_unit_dropout:
            if self.ops is not None:
                h = self.ops(h)
        return h
