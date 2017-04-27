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
    def __init__(self, unit, activation=T.tanh):
        hidden_layer.__init__(self,unit, activation)

class embedding(layer):
    def __init__(self,unit,**kwargs):
        layer.__init__(self,**kwargs)
        self.emb_dim = unit
    def init_params(self):
        self.wemb=self._allocate(uniform, 'Wemb', randn, self.in_dim, self.emb_dim)
    def apply(self):
        n_timesteps = self.input.shape[0]
        n_samples =   self.input.shape[1]
        return T.reshape(self.wemb[self.input.flatten()] ,[n_timesteps,
                                                    n_samples,self.emb_dim])

class maxout(layer):
    def __init__(self,num_pieces):
        layer.__init__(self)
        self.num_pieces=num_pieces

    def apply(self):
        last_dim = self.input.shape[-1]
        output_dim = last_dim // self.num_pieces
        new_shape = ([self.input.shape[i] for i in range(self.input.ndim - 1)] +
                     [output_dim, self.num_pieces])
        return T.max(self.input.reshape(new_shape, ndim=self.input.ndim + 1),
                            axis=self.input.ndim)

class direct(layer):
    ''' setup direct output layer inherited from base output layer '''
    def __init__(self,**kwargs):
        layer.__init__(self)
        self.cost_function = mean_square
        self.cost = None
        self.predict = None
        self.error = None
        self.setattr('cost_function')
        self.setattr('cost')
        self.setattr('predict')
        self.setattr('error')
    def get_predict(self):
        self.predict=T.round(self.output)
    def get_cost(self,Y):
        self.cost=self.cost_function(Y,self.output)
    def get_error(self,Y):
        self.error=T.mean(T.neq(Y,self.predict))

class logistic(output_layer):
    def __init__(self,unit,Activation=T.nnet.sigmoid):
       output_layer.__init__(self,unit, Activation)
       self.cost_function=cross_entropy

class softmax(output_layer):
    def __init__(self, unit, activation=T.nnet.softmax):
        output_layer.__init__(self,unit, activation)

    def get_predict(self):
        self.predict = T.argmax(self.output, axis=1)

    def get_cost(self, Y):
        self.cost = T.mean(T.nnet.categorical_crossentropy(self.output, Y))

    def get_error(self, Y):
        self.error = T.mean(T.neq(self.predict, Y))

class readout(hidden_layer):
    def __init__(self, unit, activation=T.tanh):
        hidden_layer.__init__(self,unit, activation)