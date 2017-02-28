# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 14:30:08 2017

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from Layers import Output_Layer, Layer_Tools

''' setup softmax output layer inherited from base output layer '''


class layer(Output_Layer):
    def __init__(self, Rng, N_in, N_out, Name='undefined',Wt=None, Bi=None, Wt_init='zeros',
                 Bi_init='zeros', Cost_func='neglog', Activation=T.nnet.softmax):
        Output_Layer.__init__(self, Rng, N_in, N_out,Name, Wt, Bi, Wt_init, Bi_init,
                              Cost_func, Activation)

    def init_wt_bi(self):
        Output_Layer.init_wt_bi(self)
        self.save_grad_mask=theano.shared(np.zeros((self.N_in, self.N_units),dtype='float32'),name='save_grad_mask_'+self.Name,borrow=True)

    def predict(self):
        self.pred_Y = T.argmax(self.outputs, axis=1)

    def cost(self, Y):
        return -T.mean(T.log(self.outputs)[T.arange(Y.shape[0]),Y])

    def error(self, Y):
        return T.mean(T.neq(self.pred_Y, Y))


