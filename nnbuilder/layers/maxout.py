# -*- coding: utf-8 -*-
"""
Created on  Feb 16 1:27 AM 2017

@author: aeloyq
"""


import numpy as np
import theano.tensor as T
from layers import hidden_layer
import direct
base=direct.get

''' setup hidden layer of feedforward network inherited from Hidden_Layer '''

class get(base):
    def __init__(self,num_pieces):
        base.__init__(self)
        self.num_pieces=num_pieces
    def get_output(self):
        last_dim = self.input.shape[-1]
        output_dim = last_dim // self.num_pieces
        new_shape = ([self.input.shape[i] for i in range(self.input.ndim - 1)] +
                     [output_dim, self.num_pieces])
        self.output = T.max(self.input.reshape(new_shape, ndim=self.input.ndim + 1),
                            axis=self.input.ndim)