# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 14:30:08 2017

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from basic import output_layer, utils

''' setup softmax output layer inherited from base output layer '''


class get(output_layer):
    def __init__(self, in_dim, unit_dim, activation=T.nnet.softmax):
        output_layer.__init__(self, in_dim, unit_dim, activation)

    def get_predict(self):
        self.predict = T.argmax(self.output, axis=1)

    def get_cost(self, Y):
        self.cost = T.mean(T.nnet.categorical_crossentropy(self.output, Y))

    def get_error(self, Y):
        self.error = T.mean(T.neq(self.predict, Y))
