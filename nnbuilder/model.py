# -*- coding: utf-8 -*-
"""
Created on  Feb 13 10:24 PM 2017

@author: aeloyq
"""
import config
import layers.layers as layers
import theano
import theano.tensor as T
import numpy as np
from collections import OrderedDict
import logger


class layer_():
    def __init__(self, layer, input, name=None):
        self.layer = layer
        self.input = input
        self.name = name

    def evaluate(self, op_dict):
        if isinstance(self.input, layers.baselayer):
            inp = self.input.output
        elif isinstance(self.input, node_):
            inp = self.input.output
        else:
            inp = self.input
        self.layer.build(inp, self.name, **op_dict)


class node_:
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


from layers.layers import ops


class dropout_:
    def __init__(self, layer, dp_name=None, noise=0.5):
        self.layer = layer
        # if dp_name != None and use_noise#Todo:multi setting of noise
        self.noise = theano.shared(value=noise, name='dropout_noise%s' % layer.name, borrow=True)
        if dp_name != None:
            self.dp_name = [name[0] + '_' + ops.dropout.name + '_' + self.layer.name for name in self.dp_name]
        else:
            self.dp_name = None

    def evaluate(self):
        if self.dp_name == None:
            self.op_dict = {ops.dropout.name: {ops.dropout.use_noise: self.noise}}
        else:
            self.op_dict = {ops.dropout.name: {ops.dropout.use_noise: self.noise}}
            for name in self.layer.ops:
                self.layer.ops[name] = False
            for name in self.dp_name:
                self.layer.ops[name] = True
                self.op_dict[name] = {ops.dropout.use_noise: self.noise}


class weight_decay_:
    def __init__(self, layers, params, noise=0.0001):
        self.layers = layers
        self.l2 = noise
        self.params = params

    def evaluate(self, cost):
        reg = 0
        params=[]
        if self.params == None:
            for name,layer in self.layers.items():
                for pname, param in layer.params.items():
                    if layer.roles[pname] is layers.roles.weight:
                        params.append(param)
        else:
            for layer in self.layers:
                for name, param in self.params.items():
                    if name in layer.params:
                        if layer.roles[name] == layers.roles.weight:
                            params.append(param)
        for param in params:
            reg += (param ** 2).sum()
        return cost + self.l2 * reg


class residual_:
    def __init__(self, layer, pre_layer):
        self.layer = layer
        self.pre_layer = pre_layer
        self.op_dict = {}

    def evaluate(self):
        self.op_dict['pre_tvar'] = self.pre_layer.output


class model():
    def __init__(self):
        self.X = T.matrix('X')
        self.Y = T.ivector('Y')
        self.inputs = [self.X, self.Y]
        self.train_inputs = [self.X, self.Y]
        self.model_inputs = [self.X, self.Y]
        self.rng = config.rng
        self.layers = OrderedDict()
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
        self.trng = config.trng
        self.X_mask = None
        self.Y_mask = None
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
                if isinstance(node, layer_):
                    op_dict = {}
                    for op in self.ops[node.layer]:
                        op.evaluate()
                        op_dict.update(op.op_dict)
                    op_dict['mode'] = 'use'
                    node.evaluate(op_dict)
                    self.params.update(node.layer.params)
                    self.roles.update(node.layer.roles)
                    self.raw_output = node.layer
                if isinstance(node, node_):
                    node.evaluate()
            self.raw_output.get_cost(self.Y)
            self.raw_output.get_predict()
            self.raw_output.get_error(self.Y)
            self.raw_cost = self.raw_output.cost
            self.predict = self.raw_output.predict
            self.error = self.raw_output.error

        def train():
            for node in self.graph:
                if isinstance(node, layer_):
                    op_dict = {}
                    for op in self.ops[node.layer]:
                        op.evaluate()
                        op_dict.update(op.op_dict)
                    op_dict['mode'] = 'train'
                    node.evaluate(op_dict)
                    self.output = node.layer
                if isinstance(node, node_):
                    node.evaluate()
            self.output.get_cost(self.Y)
            self.cost = self.output.cost

        raw()
        train()
        for ops in self.ops['cost']:
            self.cost = ops.evaluate(self.cost)
        for key in self.layers:
            self.user_debug_stream.extend(self.layers[key].debug_stream)
            self.updates.update(self.layers[key].updates)

    def addlayer(self, layer, input, name=None):
        self.n_layers += 1
        if name == None: name = 'layer{}'.format(self.n_layers)
        layer_instance = layer_(layer, input, name)
        self.layers[name] = layer
        self.ops[layer] = []
        self.graph.append(layer_instance)

    def addnode(self, operation, layer1, layer2, name=None):
        self.n_nodes += 1
        if name == None: name = 'layer{}'.format(self.n_nodes)
        node_instance = node_(operation, layer1, layer2, name)
        self.nodes[name] = node_instance
        self.graph.append(node_instance)

    def add_dropout(self, layer, dp_name=None, use_noise=0.5):
        drop_out_instance = dropout_(layer, dp_name, use_noise)
        self.ops[layer].append(drop_out_instance)

    def add_weight_decay(self, l2=0.0001, layers=None, params=None):
        if layers == None: layers = self.layers
        weight_decay_instance = weight_decay_(layers, params, l2)
        self.ops['cost'].append(weight_decay_instance)

    def add_residual(self, layer, pre_layer):
        residual_instance = residual_(layer, pre_layer)
        self.ops[layer].append(residual_instance)
