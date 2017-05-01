# -*- coding: utf-8 -*-
"""
Created on  四月 26 21:11 2017

@author: aeloyq
"""

from nnbuilder import config
from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T


# weights init function


def numpy_asarray_floatx(data):
    data2return = np.asarray(data, dtype=theano.config.floatX)
    return data2return


def randn(*args):
    param = config.rng.randn(*args)
    return numpy_asarray_floatx(param)


def uniform(*args):
    param = config.rng.uniform(low=-np.sqrt(6. / sum(args)), high=np.sqrt(6. / sum(args)), size=args)
    return numpy_asarray_floatx(param)


def zeros(*args):
    shape = []
    for dim in args:
        shape.append(dim)
    param = np.zeros(shape)
    return numpy_asarray_floatx(param)


def orthogonal(*args):
    param = uniform(args[0], args[0])
    param = np.linalg.svd(param)[0]
    for _ in range(args[1] / args[0] - 1):
        param_ = uniform(args[0], args[0])
        param = np.concatenate((param, np.linalg.svd(param_)[0]), 1)
    return numpy_asarray_floatx(param)


# recurrent output

def final(outputs, mask):
    return outputs[-1]


def all(outputs, mask):
    return outputs


def mean_pooling(outputs, mask):
    return ((outputs * mask[:, :, None]).sum(axis=0)) / mask.sum(axis=0)[:, None]


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = T.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = T.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out


# cost function

def mean_square(Y, output):
    return T.mean(T.square(Y - output))/2


def neg_log(Y, output):
    return T.mean(T.nnet.categorical_crossentropy(Y,output))


def cross_entropy(Y, output):
    return T.mean(T.nnet.categorical_crossentropy(Y,output))


def errors(Y_reshaped, pred_Y):
    '''
    calculate the error rates
    :param Y_reshaped: 
    :param pred_Y: 
    :return: 
    '''
    return T.mean(T.neq(pred_Y, Y_reshaped))
