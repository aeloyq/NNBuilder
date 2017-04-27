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


class node_:
    '''
    deprecated
    '''

    def __init__(self, operation, id1=None, id2=None, name=None):
        self.operation = operation
        self.id1 = id1
        self.id2 = id2
        self.name = name
        self.output = None

    def evaluate(self):
        if isinstance(self.id1, layers.baselayer):
            self.id1 = self.id1.output
        else:
            self.id1 = self.id1
        if isinstance(self.id2, layers.baselayer):
            self.id2 = self.id2.output
        else:
            self.id2 = self.id2
        if self.operation == '+':
            self.output = self.id1 + self.id2
        elif self.operation == '-':
            self.output = self.id1 - self.id2
        elif self.operation == '*':
            self.output = self.id1 * self.id2
        elif self.operation == '&':
            self.output = T.concatenate([self.id1, self.id2, self.id1.ndim - 1])


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


class model():
    def __init__(self,dim,X=X,Y=Y,**kwargs):
        self.X = X('X')
        self.Y = Y('Y')
        for name in kwargs:
            setattr(self, name, kwargs[name](name))
        self.X_dim=dim
        self.pre_dim=dim
        self.pre_layer=None
        self.inputs = [self.X, self.Y]
        self.train_inputs = [self.X, self.Y]
        self.model_inputs = [self.X, self.Y]
        self.rng = config.rng
        self.layers = OrderedDict()
        self.layers_in_dim_dict=OrderedDict()
        self.n_layers = 0
        self.nodes = OrderedDict()
        self.n_nodes = 0
        self.ops = OrderedDict()
        self.ops['cost'] = []
        self.n_ops = 0
        self.output = None
        self.raw_output = None
        self.cost = None
        self.raw_cost = None
        self.error = None
        self.graph = []
        self.params = OrderedDict()
        self.roles = OrderedDict()
        self.user_debug_stream = []
        self.X_mask = X('X_mask')
        self.Y_mask = Y('Y_mask')
        self.updates = OrderedDict()

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
            for node in self.graph:
                op_dict = {}
                for op in self.ops[node.layer]:
                    op.evaluate()
                    op_dict.update(op.op_dict)
                op_dict['mode'] = 'use'
                node.evaluate(op_dict)
                self.params.update(node.layer.params)
                self.roles.update(node.layer.roles)
                self.raw_output = node.layer
            self.raw_output.get_cost(self.Y)
            self.raw_output.get_predict()
            self.raw_output.get_error(self.Y)
            self.raw_cost = self.raw_output.cost
            self.predict = self.raw_output.predict
            self.error = self.raw_output.error

        def train():
            for node in self.graph:
                op_dict = {}
                for op in self.ops[node.layer]:
                    op.evaluate()
                    op_dict.update(op.op_dict)
                op_dict['mode'] = 'train'
                node.evaluate(op_dict)
                self.output = node.layer
            self.output.get_cost(self.Y)
            self.cost = self.output.cost

        raw()
        train()
        for ops in self.ops['cost']:
            self.cost = ops.evaluate(self.cost)
        for key in self.layers:
            self.user_debug_stream.extend(self.layers[key].debug_stream)
            self.updates.update(self.layers[key].updates)

    def add(self,element,name=None):
        if isinstance(element,basic.baselayer):
            self.n_layers += 1
            if name == None: name = 'layer{}'.format(self.n_layers)
            self.layers[name] = element
            self.pre_layer=element
            self.layers_in_dim_dict=self.pre_dim
            self.ops[element] = []
            self.graph.append(element)
        else:
            element.init(self.pre_layer,self.ops)
