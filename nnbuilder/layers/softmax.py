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


class get(output_layer):
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

class get_sequence(output_layer):
    def __init__(self, in_dim, unit_dim,activation=T.nnet.softmax):
        output_layer.__init__(self, in_dim, unit_dim,activation)
        self.cost_function=self.cost_functions.neglog

    def set_mask(self,tvar):
        self.mask=tvar

    def get_output(self):
        def _step(x_):
            out=T.nnet.nnet.softmax(T.dot(x_,self.wt)+self.bi)
            return  out
        out,upd=theano.scan(_step,sequences=[self.input],name='Softmax_sequence_scan_'+self.name,n_steps=self.input.shape[0])
        self.output=out
        self.predict()

    def predict(self):
        self.pred_Y = T.argmax(self.output, axis=2)

    def cost(self, Y):
        axis0 = T.repeat(T.arange(Y.shape[0]), Y.shape[1])
        axis1 = T.tile(T.arange(Y.shape[1]), Y.shape[0])
        axis2 = T.flatten(Y)
        mask = T.flatten(self.mask)
        flattened_0=self.output[axis0,axis1,axis2]
        flattened_1 =-T.log(flattened_0)
        flattened_2 = flattened_1*mask
        flattened_3 =T.sum(flattened_2)
        return flattened_3 / T.sum(mask)


    def error(self, Y):
        return T.mean(T.neq(self.pred_Y, Y))



