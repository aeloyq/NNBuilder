# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:55:42 2016

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from basic import *
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
        layer.__init__(self,unit,**kwargs)
        self.emb_dim = unit
    def init_params(self):
        self.wemb=self.allocate(randn, 'Wemb', weight, self.in_dim, self.emb_dim)
    def apply(self, X, P):
        n_timesteps = X.shape[0]
        n_samples =   X.shape[1]
        return T.reshape(P['Wemb'][X.flatten()], [n_timesteps,
                                                  n_samples, self.emb_dim])

class lookuptable(embedding):
    def __init__(self,unit,**kwargs):
        embedding.__init__(self,unit,**kwargs)

class maxout(baselayer):
    def __init__(self,num_pieces):
        baselayer.__init__(self)
        self.num_pieces=num_pieces

    def apply(self,X,P):
        last_dim = X.shape[-1]
        output_dim = last_dim // self.num_pieces
        new_shape = ([X.shape[i] for i in range(X.ndim - 1)] +
                     [output_dim, self.num_pieces])
        return T.max(X.reshape(new_shape, ndim=X.ndim + 1),
                            axis=X.ndim)

class compass(baselayer):
    def __init__(self,ndim=2):
        baselayer.__init__(self)
        self.n_dim=ndim

    def apply(self,X,P):
        if self.n_dim==1:
            self.shape=X.shape
            new_shape=self.shape[0]
            for i in range(1,X.ndim):
                new_shape = new_shape*self.shape[i]
            return X.reshape([new_shape])
        elif X.ndim>self.n_dim:
            self.shape = X.shape
            new_shape_1 = self.shape[0]
            for i in range(1, X.ndim-self.n_dim+1):
                new_shape_1 = new_shape_1 * self.shape[i]
            new_shape_2 = []
            for i in range(-self.n_dim+1,0):
                new_shape_2.append(self.shape[i])
            return X.reshape([new_shape_1]+new_shape_2)

class direct(baselayer):
    ''' setup direct output layer inherited from base output layer '''
    def __init__(self,**kwargs):
        baselayer.__init__(self,**kwargs)
        self.cost_function = mean_square
        self.cost = None
        self.predict = None
        self.error = None
        self.mask=False
        self.setattr('cost_function')
        self.setattr('cost')
        self.setattr('predict')
        self.setattr('error')
    def apply(self,X,P):
        return X
    def get_predict(self):
        self.predict=T.round(self.output)
    def get_cost(self,Y):
        self.cost=self.cost_function(Y,self.output)
    def get_error(self,Y):
        self.error=T.mean(T.neq(Y,self.predict))

class logistic(output_layer):
    def __init__(self,unit,**kwargs):
       output_layer.__init__(self,unit, T.nnet.sigmoid,**kwargs)
       self.cost_function=cross_entropy

class softmax(output_layer):
    def __init__(self, unit,**kwargs):
        output_layer.__init__(self,unit, T.nnet.softmax,**kwargs)

    def set_children(self):
        self.children['lb'] = linear_bias(self.unit_dim, self.activation)
        self.children['compass']=compass(2)

    def apply(self, X, P):
        shape=X.shape
        ndim=X.ndim
        if X.ndim >2:
            X=self.children['compass'].feedforward(X, P)
            output=self.children['lb'].feedforward(X, P)
            self.cost_output=output
            new_shape=[]
            for i in range(0,ndim-1):
                new_shape.append(shape[i])
            new_shape.append(output.shape[-1])
            return output.reshape(new_shape)
        else:
            self.cost_output = self.children['lb'].feedforward(X, P)
            return self.cost_output

    def get_predict(self):
        self.predict = T.argmax(self.output, axis=-1)

    def get_cost(self, Y):
        if Y.ndim >= 2:
            self.cost = T.mean(T.nnet.categorical_crossentropy(self.cost_output, Y.flatten()))
        else:
            self.cost = T.mean(T.nnet.categorical_crossentropy(self.output, Y))

    def get_error(self, Y):
        self.error = T.mean(T.neq(self.predict, Y))

class readout(hidden_layer):
    def __init__(self, unit, activation=T.tanh,**kwargs):
        hidden_layer.__init__(self,unit, activation,**kwargs)