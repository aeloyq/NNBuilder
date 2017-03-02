# -*- coding: utf-8 -*-
"""
Created on Thu Feb 09 14:30:08 2017

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T
from Layers import Output_Layer, Layer_Tools,baselayer

''' setup softmax output layer inherited from base output layer '''


class layer(baselayer):
    def __init__(self,Rng,Name='undefined',Cost_func='neglog'):
        self.Rng=Rng
        self.Cost_func=Cost_func
        self.params = []
        self.Name=Name
    def set_inputs(self,Inputs_X):
        self.Inputs=Inputs_X
        self.output_func()
    def set_name(self,name):
        self.Name=name
    def predict(self):
        self.pred_Y = T.round(self.outputs)
    def output_func(self):
        self.outputs=self.Inputs
    def cost(self, Y):
        if Y.ndim==2:
            return Layer_Tools.cost(Y,self.Cost_func,self.outputs)
        if Y.ndim==1:
            return Layer_Tools.cost(T.reshape(Y,[Y.shape[0],1]),self.Cost_func,self.outputs)

    def error(self, Y):
        if Y.ndim == 1:
            return Layer_Tools.errors(T.reshape(Y,[Y.shape[0],1]),self.pred_Y)
        if Y.ndim==2:
            return Layer_Tools.errors(Y, self.pred_Y)


