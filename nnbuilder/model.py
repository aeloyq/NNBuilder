# -*- coding: utf-8 -*-
"""
Created on  Feb 13 10:24 PM 2017

@author: aeloyq
"""
import config
import layers.basic as basic
import theano
import theano.tensor as T
import numpy as np
import logger
from collections import OrderedDict
from layers.roles import *
from layers.utils import *
from layers.ops import *



X = T.matrix
Y = T.ivector
Int4dX = T.itensor4
Int4dY = T.itensor4
Int3dX = T.itensor3
Int3dY = T.itensor3
Int2dX = T.imatrix
Int2dY = T.imatrix
IntX=T.ivector
IntY=T.ivector
Float4dX = T.ftensor4
Float4dY = T.ftensor4
Float3dX = T.ftensor3
Float3dY = T.ftensor3
Float2dX = T.fmatrix
Float2dY = T.fmatrix
FloatX=T.fvector
FloatY=T.fvector
Int4dMask=T.wtensor4
Int3dMask=T.wtensor3
Int2dMask=T.wmatrix
IntMask=T.wvector
Float4dMask=T.ftensor4
Float3dMask=T.ftensor3
Float2dMask=T.fmatrix
FloatMask=T.fvector

class model():
    def __init__(self,dim,X=X,Y=Y,**kwargs):
        self.X = X('X')
        self.Y = Y('Y')
        for name in kwargs:
            setattr(self, name, kwargs[name](name))
        self.X_dim=dim
        self.pre_dim=dim
        self.pre_layer=self.X
        self.inputs = [self.X, self.Y]
        self.rng = config.rng
        self.layers = OrderedDict()
        self.layers_in_dim_dict=OrderedDict()
        self.layers_input_dict = OrderedDict()
        self.n_layers = 0
        self.ops = OrderedDict()
        self.ops['cost'] = []
        self.n_ops = 0
        self.output = None
        self.raw_output = None
        self.cost = None
        self.raw_cost = None
        self.error = None
        self.params = OrderedDict()
        self.roles = OrderedDict()
        self.user_debug_stream = []
        self.X_mask = X('X_mask')
        self.Y_mask = Y('Y_mask')
        self.updates = OrderedDict()
        self.raw_updates = OrderedDict()

    def set_inputs(self, inputs):
        self.inputs = inputs
        self.train_inputs = inputs
        self.X = inputs[0]
        self.Y = inputs[1]

    def set_output(self, tvar):
        self.output = tvar

    def set_predict(self, tvar):
        self.predict = tvar

    def set_cost(self, tvar):
        self.cost = tvar

    def set_error(self, tvar):
        self.error = tvar

    def build(self):
        def raw():
            for name,node in self.layers.items():
                op_dict = {}
                for op in self.ops[node]:
                    op.evaluate()
                    op_dict.update(op.op_dict)
                op_dict['mode'] = 'use'
                node.evaluate(self.layers_input_dict[name],name,**op_dict)
                self.params.update(node.params)
                self.roles.update(node.roles)
                self.raw_output = node
            self.raw_output.get_cost(self.Y)
            self.raw_output.get_predict()
            self.raw_output.get_error(self.Y)
            self.raw_cost = self.raw_output.cost
            self.predict = self.raw_output.predict
            self.error = self.raw_output.error
            for key in self.layers:
                self.raw_updates.update(self.layers[key].raw_updates)

        def train():
            for name, node in self.layers.items():
                op_dict = {}
                for op in self.ops[node]:
                    op.evaluate()
                    op_dict.update(op.op_dict)
                op_dict['mode'] = 'train'
                node.evaluate(self.layers_input_dict[name],name,**op_dict)
                self.output = node
            self.output.get_cost(self.Y)
            self.cost = self.output.cost
            for key in self.layers:
                self.updates.update(self.layers[key].updates)

        train()
        raw()
        for ops in self.ops['cost']:
            self.cost = ops.evaluate(self.cost,self.layers)
        for key in self.layers:
            self.user_debug_stream.extend(self.layers[key].debug_stream)

    def add(self,element,name=None):
        if isinstance(element,basic.baselayer):
            self.n_layers += 1
            if name == None: name = 'layer{}'.format(self.n_layers)
            self.layers[name] = element
            element.set_in_dim(self.pre_dim)
            self.layers_input_dict[name]=self.pre_layer
            self.ops[element] = []
            self.pre_layer=element
            if self.X_mask in self.inputs:element.x_mask=self.X_mask
            if self.Y_mask in self.inputs:element.y_mask=self.Y_mask;element.y=self.Y
            if hasattr(element,'unit_dim'):
                self.pre_dim=element.unit_dim
        else:
            element.init(self.pre_layer, self.ops)

    def sequential(self,X=Int2dMask,Y=None):
        if X is not None: self.X_mask=X('X_mask');self.inputs.append(self.X_mask)
        if Y is not None: self.Y_mask=Y('Y_mask');self.inputs.append(self.Y_mask)

