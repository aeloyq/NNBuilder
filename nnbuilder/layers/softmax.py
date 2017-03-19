# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 14:30:08 2017

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from layers import output_layer, layer_tools

''' setup softmax output layer inherited from base output layer '''


class get_new(output_layer):
    def __init__(self, in_dim, unit_dim,activation=T.nnet.softmax):
        output_layer.__init__(self, in_dim, unit_dim,activation)
        self.cost_function=self.cost_functions.neglog
        self.masked_y=False

    def predict(self):
        self.pred_Y = T.argmax(self.output, axis=self.output.ndim-1)

    def cost(self, Y):
        if Y.ndim==1:
            return self.cost_function(Y,self.output)

        if Y.ndim==2:
            axis0 = T.tile(T.arange(Y.shape[1]), Y.shape[0])
            axis1 = T.repeat(np.arange(T.shape[0]), T.shape[1])
            axis2 = T.reshape(Y, Y.shape[0] * Y.shape[1])
            if not self.masked_y:
                return -T.mean(T.log(self.output)[axis0,axis1,axis2])
            else:
                mask=T.reshape(self.Y_mask, self.Y_mask.shape[0] * self.Y_mask.shape[1])
                return -T.log(self.output)[axis0,axis1,axis2]*mask/T.sum(mask)


    def error(self, Y):
        if Y.ndim==1:
            return T.mean(T.neq(self.pred_Y, Y))

        if Y.ndim==2:
            if not self.masked_y:
                return T.mean(T.neq(self.pred_Y, Y))
            else:
                return T.mean(T.neq(self.pred_Y, Y))



