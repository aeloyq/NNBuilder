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
        return -T.log(self.output[-1,T.arange(Y.shape[0]), Y]).mean()


    def error(self, Y):
        return T.mean(T.neq(self.pred_Y, Y))

class get_sequence(output_layer):
    def __init__(self, in_dim, unit_dim,activation=T.nnet.softmax):
        output_layer.__init__(self, in_dim, unit_dim,activation)
        self.cost_function=self.cost_functions.neglog
        self.wt1='wt1'
        self.wt2='wt2'
        self.params=[self.wt1,self.wt2]
        self.param_init_function={'wt1':self.param_init_functions.uniform,
                                  'wt2': self.param_init_functions.uniform}

    def init_layer_params(self):
        self.n_classes = int(np.ceil(np.sqrt(self.unit_dim)))
        wt1_values =self.param_init_function[self.wt1](self.in_dim,self.n_classes)
        wt2_values = self.param_init_function[self.wt2](self.n_classes,self.in_dim,self.n_classes)
        bi1_values=np.zeros(self.n_classes).astype('float32')
        bi2_values = np.zeros([self.n_classes,self.n_classes]).astype('float32')
        self.wt1 = theano.shared(value=wt1_values, name='Hierarchical_Softmax_Wt1' + '_' + self.name, borrow=True)
        self.wt2 = theano.shared(value=wt2_values, name='Hierarchical_Softmax_Wt2' + '_' + self.name, borrow=True)
        self.bi1 = theano.shared(value=bi1_values, name='Hierarchical_Softmax_Bi1' + '_' + self.name, borrow=True)
        self.bi2 = theano.shared(value=bi2_values, name='Hierarchical_Softmax_Bi2' + '_' + self.name, borrow=True)
        self.params=[self.wt1,self.wt2]

    def set_mask(self,tvar):
        self.mask=tvar

    def get_output(self):
        def _step(x_):
            out=T.nnet.h_softmax(x_,x_.shape[0],self.n_classes*self.n_classes,self.n_classes,self.n_classes,self.wt1,self.bi1,self.wt2,self.bi2)
            return  out
        out,upd=theano.scan(_step,sequences=[self.input],name='Softmax_sequence_scan_'+self.name,n_steps=self.input.shape[0])
        self.output=out
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



