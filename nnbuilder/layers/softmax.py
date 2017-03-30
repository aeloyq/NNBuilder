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

    def predict(self):
        self.pred_Y = T.argmax(self.output, axis=self.output.ndim-1)

    def cost(self, Y):
        return -T.log(self.output[T.arange(Y.shape[0]), Y] + 1e-6).mean()


    def error(self, Y):
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

    def get_output_(self):

        out=T.exp(T.dot(self.input,self.wt)+self.bi)
        norm=T.sum(out,0).dimshuffle(0,1,'x')
        self.output=out/norm

        self.predict()

    def predict(self):
        self.pred_Y = T.argmax(self.output, axis=2)


    def cost(self, Y):
        def _step(o,y,m):
            return -T.mean(T.log(o)[T.arange(y.shape[0]), y]*m)
        c,u=theano.scan(_step,[self.output,Y,self.mask],n_steps=self.mask.shape[0])
        return T.mean(c)



    def error(self, Y):
        return T.mean(T.neq(self.pred_Y, Y))



