# -*- coding: utf-8 -*-
"""
Created on  Feb 13 10:24 PM 2017

@author: aeloyq
"""
import numpy as np
import logger
import copy
from collections import OrderedDict
from layers.basic import base, entity
from layers.roles import *
from layers.utils import *
from layers.ops import *
from nnbuilder.kernel import *


class var:
    class X:
        default = [batch, feature], kernel.config.floatX
        sequence = [time, batch], kernel.config.catX
        image = [batch, channel, height, width], kernel.config.floatX

    class Y:
        default = [batch], kernel.config.catX
        numerical = [batch], kernel.config.floatX
        catglory = [batch], kernel.config.catX
        image = [batch], kernel.config.catX
        sequence = [time, batch], kernel.config.catX


class input_layer(base):
    def __init__(self, **kwargs):
        super(input_layer, self).__init__(**kwargs)
        self.name = 'input layer'

    def set_input_layer(self, input, dim):
        self.set_in_dim(dim)
        self.set_input(input)
        self.get_out_dim()
        self.output = input


class model(object):
    def __init__(self, dim, X=var.X.default, Y=var.Y.default):
        self.X_dim = dim
        self.varX = X
        self.varY = Y
        self.X = kernel.placeholder('X', X[0], X[1])
        self.Y = kernel.placeholder('Y', Y[0], Y[1])
        self.X_mask = None
        self.Y_mask = None
        self.inputs = [self.X, self.Y]
        self.data = {'X': self.X, 'Y': self.Y, 'X_Mask': None, 'Y_Mask': None}
        self.input_layer = input_layer()
        self.input_layer.set_input_layer(self.X, dim)
        #
        self.layers = OrderedDict()
        self.n_layer = 0
        self.ops = OrderedDict()
        self.lossops = []
        self.ops_options = OrderedDict()
        self.last_added_layer = self.input_layer
        #
        self.output_layer = None
        self.output = None
        self.raw_output = None
        self.loss = None
        self.sample = None
        self.sample_loss = None
        self.sample_error = None
        self.predict = None
        self.trainable_params = OrderedDict()
        self.roles = OrderedDict()
        self.shapes = OrderedDict()
        self.rndfns = OrderedDict()
        self.untrainable_params = OrderedDict()
        self.updates = OrderedDict()
        self.raw_updates = OrderedDict()
        self.user_debug_stream = []
        #
        self.intX = False
        self.intY = False
        self.seqX = False
        self.seqY = False
        if X[1] in [kernel.config.catX, kernel.config.boolX, kernel.config.intX]:
            self.intX = True
        if Y[1] in [kernel.config.catX, kernel.config.boolX, kernel.config.intX]:
            self.intY = True

    def build(self, raw_only=False):
        for name, layer in self.layers.items():
            for op in self.ops[layer]:
                op.build(layer, self.ops_options[layer][op.name])

        def train():
            for name, layer in self.layers.items():
                ops_option = copy.deepcopy(self.ops_options[layer])
                ops_option['mode'] = 'train'
                layer.build(ops_option)
                self.updates.update(layer.updates)
            self.output_layer = self.layers.values()[-1]
            self.output = self.output_layer.output
            self.loss = self.output_layer.apply_loss(self.Y)
            self.output_layer.loss = self.loss

        def raw():
            for name, layer in self.layers.items():
                ops_option = copy.deepcopy(self.ops_options[layer])
                ops_option['mode'] = 'use'
                layer.build(ops_option)
                self.trainable_params.update(layer.trainable_params)
                self.untrainable_params.update(layer.untrainable_params)
                self.roles.update(layer.roles)
                self.shapes.update(layer.shapes)
                self.rndfns.update(layer.rndfns)
                layer.raw_output = layer.output
                self.raw_updates.update(layer.raw_updates)
                self.user_debug_stream.extend(layer.debug_stream)
            self.output_layer = self.layers.values()[-1]
            self.raw_output = self.output_layer.output
            self.sample = self.output_layer.apply_sample()
            self.output_layer.sample = self.sample
            self.sample_loss = self.output_layer.apply_sample_loss(self.Y)
            self.output_layer.sample_loss = self.sample_loss
            self.sample_error = self.output_layer.apply_sample_error(self.Y)
            self.output_layer.sample_error = self.sample_error
            self.predict = self.output_layer.apply_predict()
            self.output_layer.predict = self.predict

        if not raw_only:
            train()
        raw()
        for ops in self.lossops:
            self.loss = ops.build(self.loss)

    def add(self, element, name=None):
        if isinstance(element, entity):
            self.n_layer += 1
            if name == None: name = 'layer{}'.format(self.n_layer)
            self.layers[name] = element
            element.pre_build(name, self.last_added_layer, self.data)
            self.last_added_layer = element
            self.ops[element] = []
            self.ops_options[element] = OrderedDict()

        elif isinstance(element, inlayerops):
            self.ops[self.last_added_layer].append(element)
            if element.name not in self.ops_options[self.last_added_layer]:
                self.ops_options[self.last_added_layer][element.name] = OrderedDict()
            element.init(self.last_added_layer)

        elif isinstance(element, lossops):
            self.lossops.append(element)
            element.init(self.last_added_layer, self)

        else:
            if isinstance(element, base):
                raise TypeError('Please use entity layer class !')

    def sequential(self, X=True, Y=False):
        if X:
            self.X_mask = kernel.placeholder('X_Mask', self.varX[0], kernel.config.floatX)
            self.inputs.append(self.X_mask)
            self.data['X_Mask'] = self.X_mask
            self.seqX = True
        if Y:
            self.Y_mask = kernel.placeholder('Y_Mask', self.varY[0], kernel.config.floatX)
            self.inputs.append(self.Y_mask)
            self.data['Y_Mask'] = self.Y_mask
            self.seqY = True
