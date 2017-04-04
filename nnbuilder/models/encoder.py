# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 14:30:08 2017

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from nnbuilder.layers.layers import hidden_layer, layer_tools, baselayer, costfunctions
import nnbuilder.layers.lstm, nnbuilder.layers.gru
from nnbuilder.layers.recurrent import output_ways

''' setup softmax output layer inherited from base output layer '''


class get_bi_lstm(baselayer):
    def __init__(self, in_dim, unit_dim, h_0_init=False, c_0_init=False, activation=None, **kwargs):
        baselayer.__init__(self)
        self.forward = nnbuilder.layers.lstm.get(in_dim, unit_dim, h_0_init, c_0_init, activation, **kwargs)
        self.forward.masked=False
        self.backward = nnbuilder.layers.lstm.get(in_dim, unit_dim, h_0_init, c_0_init, activation, **kwargs)
        self.backward.masked=False

    def init_layer_params(self):
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

class get_bi_gru(baselayer):
    def __init__(self, in_dim, unit_dim, h_0_init=False, activation=None, **kwargs):
        baselayer.__init__(self)
        self.forward = nnbuilder.layers.gru.get(in_dim, unit_dim, h_0_init, activation, **kwargs)
        self.forward.masked=False
        self.backward = nnbuilder.layers.gru.get(in_dim, unit_dim, h_0_init, activation, **kwargs)
        self.backward.masked=False

    def init_layer_params(self):
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
