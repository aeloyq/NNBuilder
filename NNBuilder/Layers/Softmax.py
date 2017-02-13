# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 14:30:08 2017

@author: aeloyq
"""

import numpy as np
import theano.tensor as T
from Layers import Output_Layer, Layer_Tools

''' setup softmax output layer inherited from base output layer '''


class layer(Output_Layer):
    def __init__(self, Rng, N_in, N_out, l0=0., l1=0., l2=0., Wt=None, Bi=None, Wt_init='uniform',
                 Bi_init='zeros', Hidden_Layer_Struct=[], Cost_func='square', Activation=T.nnet.softmax):
        Output_Layer.__init__(self, Rng, N_in, N_out, l0, l1, l2, Wt, Bi, Wt_init, Bi_init,
                              Hidden_Layer_Struct,
                              Cost_func, Activation)

    def predict(self):
        self.pred_Y = T.argmax(self.outputs, axis=1)

    def cost(self, Y):
        return Layer_Tools.cost(np.array(1, dtype='float32'), self.Cost_func, self.outputs[T.arange(Y.shape[0]), Y],
                                self.wt_packs, self.l0, self.l1, self.l2)

    def error(self, Y):
        return T.mean(T.neq(self.pred_Y, Y))
