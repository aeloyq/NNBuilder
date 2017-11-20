# -*- coding: utf-8 -*-
"""
Created on  四月 26 21:11 2017

@author: aeloyq
"""

import numpy as np
from nnbuilder.kernel import *

gated_activations = [T.glu, T.gtu, T.maxout]


class Parameter(object):
    def __init__(self, layer, name, role, random=None, shape=(), value=None, trainable=True):
        '''
        Class of Parameter in Layer
        :param layer(Layer):
            name of Parameter
        :param name(string):
            name of Parameter
        :param shape(tuple or list):
            shape of Parameter
        :param random(callable): None
            random function to initiate Parameter's value
        :param role(Role): None
            role of Parameter, ex:linear weights, bias or convolutional filters
        :param value(array-like data structure): None
            the value of Parameter
        :param trainable(bool): True
            Determine whether to train Parameter
        '''
        self.layer = layer
        self.name = self.layer.name + '_' + name
        self.random = random
        self.in_dim = None
        self.shape = shape
        self.size = shape
        self.role = role
        self.attr = self.role.attr
        self.trainable = trainable
        if value is not None:
            self.variable = kernel.shared(value=value, name=self.name, attr=self.role.attr)
            self.graph = self.variable.graph

    def __str__(self):
        return self.name

    def __call__(self, *args, **kwargs):
        return self.variable

    def __setattr__(self, key, value):
        if key == 'variable':
            self.__dict__[key] = value
            self.graph = value.graph
        else:
            self.__dict__[key] = value

    def init(self):
        '''
        initiate the Parameter
        :return: None
        '''
        if not hasattr(self, 'variable'):
            self.variable = kernel.shared(value=self.random(self.shape), name=self.name, attr=self.role.attr)
            self.graph = self.variable.graph

    def reinit(self, value=None):
        '''
        re-initiate the Parameter
        :param value(array-like data structure): None
            the value of Parameter
        :return: None
        '''
        if self.random is not None:
            self.variable = kernel.shared(value=self.random(self.shape), name=self.name, attr=self.role.attr)
        else:
            self.variable = kernel.shared(value=value, name=self.name, attr=self.role.attr)
        self.graph = self.variable.graph

    @property
    def value(self):
        '''
        get the value of Parameter
        :return: numpy.array
            the value of Parameter
        '''
        return self.variable.get()

    @staticmethod
    def numpy_asarray_floatx(data):
        data2return = np.asarray(data, dtype=kernel.config.floatX)
        return data2return

    @staticmethod
    def randn(shape):
        param = 0.01 * kernel.random.randn(*tuple(shape))
        return Parameter.numpy_asarray_floatx(param)

    @staticmethod
    def rand(shape):
        param = 0.01 * kernel.random.rand(*tuple(shape))
        return Parameter.numpy_asarray_floatx(param)

    @staticmethod
    def uniform(shape):
        param = kernel.random.uniform(low=-np.sqrt(6. / sum(shape)), high=np.sqrt(6. / sum(shape)), size=shape)
        return Parameter.numpy_asarray_floatx(param)

    @staticmethod
    def scaled_uniform(shape):
        param = kernel.random.uniform(low=-np.sqrt(6. / sum(shape)), high=np.sqrt(6. / sum(shape)), size=shape) * 4
        return Parameter.numpy_asarray_floatx(param)

    @staticmethod
    def zeros(shape):
        param = np.zeros(shape)
        return Parameter.numpy_asarray_floatx(param)

    @staticmethod
    def ones(shape):
        param = np.ones(shape)
        return Parameter.numpy_asarray_floatx(param)

    @staticmethod
    def orthogonal(shape):
        param = np.random.rand(shape[0], shape[0])
        param = np.linalg.svd(param)[0]
        for _ in range(shape[1] / shape[0] - 1):
            param_ = np.random.rand(shape[0], shape[0])
            param = np.concatenate((param, np.linalg.svd(param_)[0]), 1)
        return Parameter.numpy_asarray_floatx(param)

    @staticmethod
    def convweight(shape):
        fan_in = np.prod(shape[1:])
        fan_out = (shape[0] * np.prod(shape[2:]) // 4)
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        param = kernel.random.uniform(low=-W_bound, high=W_bound, size=shape)
        return Parameter.numpy_asarray_floatx(param)

    class Weight:
        attr = ['unit', 'unit']

        def __str__(self):
            return 'weight'

    weight = Weight()

    class Bias:
        attr = ['unit']

        def __str__(self):
            return 'bias'

    bias = Bias()

    class Scalew:
        attr = ['unit']

        def __str__(self):
            return 'scalew'

    scalew = Scalew()

    class Table:
        attr = ['vocab', 'unit']

        def __str__(self):
            return 'table'

    table = Table()

    class NormScale:
        attr = ['unit']

        def __str__(self):
            return 'normscale'

    normscale = NormScale()

    class KernelWeight:
        attr = ['unit', 'unit']

        def __str__(self):
            return 'kernelweight'

    kernelweight = KernelWeight()

    class NormBias:
        attr = ['unit']

        def __str__(self):
            return 'normweight'

    normBias = NormBias()

    class Trans:
        attr = ['unit', 'unit']

        def __str__(self):
            return 'trans'

    trans = Trans()

    class Conv2w:
        attr = ['in_channel', 'out_channel', 'hight', 'width']

        def __str__(self):
            return 'conv2w'

    conv2w = Conv2w()

    class Conv3w:
        attr = ['in_channel', 'out_channel', 'deepth', 'hight', 'width']

        def __str__(self):
            return 'conv3w'

    conv3w = Conv3w()
