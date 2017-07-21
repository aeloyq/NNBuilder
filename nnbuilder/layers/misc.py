# -*- coding: utf-8 -*-
"""
Created on  四月 27 23:10 2017

@author: aeloyq
"""
import numpy as np
import theano
import theano.tensor as T
from basic import *
from simple import *
from utils import *
from roles import *
from ops import *
from sequential import *
class rgru(sequential):
    def __init__(self, unit, mask=True, **kwargs):
        sequential.__init__(self, mask=mask, **kwargs)
        self.mask = mask
        self.out = 'all'
        self.unit_dim = unit
        self.setattr('out')
        self.setattr('condition')

    def set_children(self):
        self.children['input'] = linear_bias(self.unit_dim)
        self.children['gate'] = linear_bias(self.unit_dim * 3)

    def init_params(self):
        self.u = self.allocate(orthogonal, 'U', weight, self.unit_dim, self.unit_dim)
        self.ug = self.allocate(orthogonal, 'Ug', weight, self.unit_dim, self.unit_dim * 3)

    def prepare(self, X, P):
        self.state_before = [self.children['input'].feedforward(X),
                             self.children['gate'].feedforward(X)]
        if self.mask:
            self.state_before.append(self.x_mask)
        self.initiate_state = [T.zeros([self.n_samples, self.unit_dim], theano.config.floatX)]
        self.context = [P['U'], P['Ug']]
        self.n_steps = self.input.shape[0]

    def apply(self, X, P):
        def step(x, xg, h_, u, ug):
            gate = xg + T.dot(h_, ug)
            z_gate = T.nnet.relu(self.slice(gate, 0, self.unit_dim))+1e-8
            z_gate = self.addops('z_gate', z_gate, dropout, False)
            zp_gate = T.nnet.relu(self.slice(gate, 1, self.unit_dim))+1e-8
            zp_gate = self.addops('zp_gate', zp_gate, dropout, False)
            r_gate = T.nnet.sigmoid(self.slice(gate, 2, self.unit_dim))
            r_gate = self.addops('r_gate', r_gate, dropout, False)
            h_c = T.tanh(x + T.dot(r_gate * h_, u))
            z_gate_sum=z_gate+zp_gate
            h = (zp_gate/z_gate_sum) * h_ + (z_gate/z_gate_sum) * h_c
            h = self.addops('hidden_unit', h, dropout)
            return h

        def step_mask(x, xg, m, h_, u, ug):
            h = step(x, xg, h_, u, ug)
            return h

        if not self.mask:
            return step
        else:
            return step_mask


