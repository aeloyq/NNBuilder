# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 14:30:08 2017

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from nnbuilder.layers.basic import hidden_layer, utils, baselayer
import nnbuilder.layers.lstm, nnbuilder.layers.gru
from nnbuilder.layers import recurrent

''' setup softmax output layer inherited from base output layer '''


class get_bi_lstm(baselayer):
    def __init__(self, in_dim, unit_dim, h_0_init=False, c_0_init=False, activation=None, **kwargs):
        baselayer.__init__(self)
        self.forward = nnbuilder.layers.lstm.get(in_dim, unit_dim, h_0_init, c_0_init, activation, **kwargs)
        self.forward.masked=False
        self.backward = nnbuilder.layers.lstm.get(in_dim, unit_dim, h_0_init, c_0_init, activation, **kwargs)
        self.backward.masked=False

    def init_params(self):
        self.forward.init_layer_params()
        self.backward.init_layer_params()
        self.params = self.forward.params
        self.params.extend(self.backward.params)

    def set_name(self, name):
        self.name = name
        self.forward.name = 'Forward_' + name
        self.backward.name = 'Backward_' + name

    def set_x_mask(self, tvar):
        self.x_mask=tvar
        self.forward.set_x_mask(tvar)
        self.backward.set_x_mask(tvar[:, ::-1])

    def set_input(self, X):
        self.input = X
        self.forward.set_input(X)
        self.backward.set_input(X[:, ::-1, :])

    def get_output(self):
        self.forward.get_output()
        self.backward.get_output()
        self.output = T.concatenate([self.forward.output , self.backward.output[:, ::-1, :]],2)

class get_ff(hidden_layer):
    def __init__(self, in_dim, unit_dim, activation=T.tanh, **kwargs):
        hidden_layer.__init__(self ,in_dim, unit_dim*2, activation)

class get_bi_gru(baselayer):
    def __init__(self, in_dim, unit_dim, h_0_init=False, activation=None, **kwargs):
        baselayer.__init__(self)
        self.forward = nnbuilder.layers.gru.get(in_dim, unit_dim, h_0_init, activation, **kwargs)
        self.backward = nnbuilder.layers.gru.get(in_dim, unit_dim, h_0_init, activation, **kwargs)

    def init_params(self):
        self.forward.init_layer_params()
        self.backward.init_layer_params()
        self.params = self.forward.params
        self.params.extend(self.backward.params)

    def set_name(self, name):
        self.name = name
        self.forward.name = 'Forward_' + name
        self.backward.name = 'Backward_' + name

    def set_x_mask(self, tvar):
        self.x_mask=tvar
        self.forward.set_x_mask(tvar)
        self.backward.set_x_mask(tvar[:, ::-1])

    def set_input(self, X):
        self.input = X
        self.forward.set_input(X)
        self.backward.set_input(X[:, ::-1, :])

    def get_output(self):
        self.forward.get_output()
        self.backward.get_output()
        self.output = T.concatenate([self.forward.output , self.backward.output[:, ::-1, :]],2)
class get_bi_gru__(baselayer):
    def __init__(self, in_dim, unit_dim, h_0_init=False, activation=None, **kwargs):
        baselayer.__init__(self)
        self.forward = nnbuilder.layers.gru.get(in_dim, unit_dim, h_0_init, activation, **kwargs)
        self.backward = nnbuilder.layers.gru.get(in_dim, unit_dim, h_0_init, activation, **kwargs)

    def init_params(self):
        self.forward.init_layer_params()
        self.backward.init_layer_params()
        self.params = self.forward.params
        self.params.extend(self.backward.params)

    def set_name(self, name):
        self.name = name
        self.forward.name = 'Forward_' + name
        self.backward.name = 'Backward_' + name

    def set_x_mask(self, tvar):
        self.x_mask=tvar
        self.forward.set_x_mask(tvar)
        self.backward.set_x_mask(tvar[:, ::-1])

    def set_input(self, X):
        self.input = X
        self.forward.set_input(X)
        self.backward.set_input(X[:, ::-1, :])

    def get_output(self):
        self.forward.get_output()
        self.backward.get_output()
        self.output = T.concatenate([self.forward.output , self.backward.output[::-1, :]],2)
base=recurrent.get
class get_bi_gru_(base):
    def __init__(self, in_dim, unit_dim, h_0_init=False, activation=None, **kwargs):
        base.__init__(self, in_dim, unit_dim, h_0_init, activation, **kwargs)
        self.ug = 'ug'
        self.wg = 'wg'
        self.big = 'big'
        self.bwt='bwt'
        self.bu='bu'
        self.bbi='bbi'
        self.bug = 'bug'
        self.bwg = 'bwg'
        self.bbig = 'bbig'
        self.param_init_function['ug'] = self.param_init_functions.orthogonal
        self.param_init_function['wg'] = self.param_init_functions.uniform
        self.param_init_function['big'] = self.param_init_functions.zeros
        self.params = [self.wt, self.bi, self.wg, self.big, self.u, self.ug, self.bwt, self.bbi, self.bwg, self.bbig,
                       self.bu, self.bug]
        self.output_way = self.output_ways.all

    def init_layer_params(self):
        wt_values = self.param_init_function['wt'](self.in_dim, self.unit_dim)
        wg_values = self.param_init_function['wg'](self.in_dim, self.unit_dim * 2)
        bi_values = self.param_init_function['bi'](self.unit_dim)
        big_values = self.param_init_function['big'](self.unit_dim * 2)
        u_values = self.param_init_function['u'](self.unit_dim, self.unit_dim)
        ug_values = self.param_init_function['ug'](self.unit_dim, self.unit_dim * 2)
        self.wt = theano.shared(value=wt_values, name='Fowd_Wt' + '_' + self.name, borrow=True)
        self.bi = theano.shared(value=bi_values, name='Fowd_Bi' + '_' + self.name, borrow=True)
        self.u = theano.shared(value=u_values, name='Fowd_U' + '_' + self.name, borrow=True)
        self.wg = theano.shared(value=wg_values, name='Fowd_Wg' + '_' + self.name, borrow=True)
        self.big = theano.shared(value=big_values, name='Fowd_Big' + '_' + self.name, borrow=True)
        self.ug = theano.shared(value=ug_values, name='Fowd_Ug' + '_' + self.name, borrow=True)
        bwt_values = self.param_init_function['wt'](self.in_dim, self.unit_dim)
        bwg_values = self.param_init_function['wg'](self.in_dim, self.unit_dim * 2)
        bbi_values = self.param_init_function['bi'](self.unit_dim)
        bbig_values = self.param_init_function['big'](self.unit_dim * 2)
        bu_values = self.param_init_function['u'](self.unit_dim, self.unit_dim)
        bug_values = self.param_init_function['ug'](self.unit_dim, self.unit_dim * 2)
        self.bwt = theano.shared(value=bwt_values, name='Back_Wt' + '_' + self.name, borrow=True)
        self.bbi = theano.shared(value=bbi_values, name='Back_Bi' + '_' + self.name, borrow=True)
        self.bu = theano.shared(value=bu_values, name='Back_U' + '_' + self.name, borrow=True)
        self.bwg = theano.shared(value=bwg_values, name='Back_Wg' + '_' + self.name, borrow=True)
        self.bbig = theano.shared(value=bbig_values, name='Back_Big' + '_' + self.name, borrow=True)
        self.bug = theano.shared(value=bug_values, name='Back_Ug' + '_' + self.name, borrow=True)
        self.params = [self.wt, self.bi, self.wg, self.big, self.u, self.ug,self.bwt, self.bbi, self.bwg, self.bbig, self.bu, self.bug]

    def slice(self, _x, n, dim):
        return _x[:, n * dim:(n + 1) * dim]

    def get_output(self):
        self.get_n_samples()
        h_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), self.n_samples, self.unit_dim)
        lin_out, scan_update = theano.scan(self.step_mask, sequences=[self.input, self.x_mask,self.input[:,::-1,:],self.x_mask[:,::-1]],
                                           outputs_info=[h_0,h_0], name=self.name + '_Scan',
                                           n_steps=self.input.shape[0])
        self.output = T.concatenate([lin_out[0],lin_out[1][:,::-1,:]],2)
    def step_mask(self, x_, m_,bx_,bm_, h_,bh_):
        preact = T.dot(h_, self.ug) + T.dot(x_, self.wg)+ self.big

        r = T.nnet.sigmoid(self.slice(preact, 0, self.unit_dim))
        z = T.nnet.sigmoid(self.slice(preact, 1, self.unit_dim))

        h_c = T.tanh(T.dot(x_,self.wt)+T.dot(r*h_,self.u)+self.bi)

        h = (1-z)*h_+z*h_c

        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        bpreact = T.dot(bh_, self.bug) + T.dot(bx_, self.bwg) + self.bbig

        br = T.nnet.sigmoid(self.slice(bpreact, 0, self.unit_dim))
        bz = T.nnet.sigmoid(self.slice(bpreact, 1, self.unit_dim))

        bh_c = T.tanh(T.dot(bx_, self.bwt) + T.dot(br * bh_, self.bu) + self.bbi)

        bh = (1 - bz) * bh_ + bz * bh_c

        bh = bm_[:, None] * bh + (1. - bm_)[:, None] * bh_

        return h,bh

