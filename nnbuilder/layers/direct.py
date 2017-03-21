# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 14:30:08 2017

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from layers import output_layer, layer_tools,baselayer,costfunctions

''' setup softmax output layer inherited from base output layer '''


class get(baselayer):
    def __init__(self,**kwargs):
        baselayer.__init__(self)
        self.cost_function=layer_tools.square_cost
        self.cost_functions = costfunctions()
    def get_output(self):
        baselayer.get_output(self)
        if self.ops is not None:
            self.output = self.ops(self.output)
        self.predict()
    def predict(self):
        self.pred_Y=T.round(self.output)
    def cost(self,Y):
        if Y.ndim==2:
            return self.cost_function(Y, self.output)
        if Y.ndim==1:
            return self.cost_function(T.reshape(Y, [Y.shape[0], 1]),  self.output)
    def error(self,Y):
        if Y.ndim == 1:
            return layer_tools.errors(T.reshape(Y, [Y.shape[0], 1]), self.pred_Y)
        if Y.ndim==2:
            return layer_tools.errors(Y, self.pred_Y)


