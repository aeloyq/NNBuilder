# -*- coding: utf-8 -*-
"""
Created on  四月 26 21:11 2017

@author: aeloyq
"""

import numpy as np
import roles
from collections import OrderedDict
from nnbuilder.kernel import *


class param_init_functions:
    '''
    # Weights Init Functions
    '''

    @staticmethod
    def numpy_asarray_floatx(data):
        data2return = np.asarray(data, dtype=kernel.config.floatX)
        return data2return

    @staticmethod
    def randn(shape):
        param = 0.01 * kernel.random.randn(*tuple(shape))
        return param_init_functions.numpy_asarray_floatx(param)

    @staticmethod
    def rand(shape):
        param = 0.01 * kernel.random.rand(*tuple(shape))
        return param_init_functions.numpy_asarray_floatx(param)

    @staticmethod
    def uniform(shape):
        param = kernel.random.uniform(low=-np.sqrt(6. / sum(shape)), high=np.sqrt(6. / sum(shape)), size=shape)
        return param_init_functions.numpy_asarray_floatx(param)

    @staticmethod
    def scaled_uniform(shape):
        param = kernel.random.uniform(low=-np.sqrt(6. / sum(shape)), high=np.sqrt(6. / sum(shape)), size=shape) * 4
        return param_init_functions.numpy_asarray_floatx(param)

    @staticmethod
    def zeros(shape):
        param = np.zeros(shape)
        return param_init_functions.numpy_asarray_floatx(param)

    @staticmethod
    def ones(shape):
        param = np.ones(shape)
        return param_init_functions.numpy_asarray_floatx(param)

    @staticmethod
    def orthogonal(shape):
        param = np.random.rand(shape[0], shape[0])
        param = np.linalg.svd(param)[0]
        for _ in range(shape[1] / shape[0] - 1):
            param_ = np.random.rand(shape[0], shape[0])
            param = np.concatenate((param, np.linalg.svd(param_)[0]), 1)
        return param_init_functions.numpy_asarray_floatx(param)

    @staticmethod
    def convweight(shape, poolsize):
        fan_in = np.prod(shape[1:])
        fan_out = (shape[0] * np.prod(shape[2:]) // np.prod(poolsize))
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        param = kernel.random.uniform(low=-W_bound, high=W_bound, size=shape)
        return param_init_functions.numpy_asarray_floatx(param)


class utils:
    '''
    Misc Utils
    '''
    gated_activation = [T.glu, T.gtu, T.maxout]

    @staticmethod
    def slice(_x, n, dim):
        if _x.ndim == 2:
            return _x[:, n * dim:(n + 1) * dim]
        elif _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        elif _x.ndim == 4:
            return _x[:, :, :, n * dim:(n + 1) * dim]

    @staticmethod
    def compress(X, ndim):
        shape = X.shape
        compressed_shape = shape[0]
        for i in range(1, X.ndim - ndim + 1):
            compressed_shape = compressed_shape * shape[i]
        remained_shape = []
        for i in range(-ndim + 1, 0):
            remained_shape.append(shape[i])
        return X.reshape([compressed_shape] + remained_shape, [None] + [X.attr[-1]])

    @staticmethod
    def zeros_initiator(batch_size, unit_dim):
        return T.zeros([batch_size, unit_dim], [roles.batch, roles.unit])


class loss_functions:
    '''
    Cost Functions
    '''

    @staticmethod
    def mse(y, y_true):
        return T.mean_square(y, y_true)

    @staticmethod
    def rmse(y, y_true):
        return T.root_mean_square(y, y_true)

    @staticmethod
    def ce(y, y_true):
        return T.binary_crossentropy(y, y_true)

    @staticmethod
    def nlog(y, y_true):
        return T.categorical_crossentropy(y, y_true)
