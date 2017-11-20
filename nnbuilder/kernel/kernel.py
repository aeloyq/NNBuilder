# -*- coding: utf-8 -*-
"""
Created on  八月 03 23:44 2017

@author: aeloyq
"""
import os
import numpy as np
import copy
from collections import OrderedDict

if "NNBUILDER_FLAGS" in os.environ:
    if os.environ["NNBUILDER_FLAGS"] == 'Theano':
        import Theano as Backend
    elif os.environ["NNBUILDER_FLAGS"] == 'Pytorch':
        import Pytorch as Backend
    elif os.environ["NNBUILDER_FLAGS"] == 'Tensorflow':
        import Tensorflow as Backend
    elif os.environ["NNBUILDER_FLAGS"] == 'Mxnet':
        import Mxnet as Backend
else:
    import Theano as Backend


class Attr:
    batch = 'batch'
    unit = 'unit'
    channel = 'channel'
    deepth = 'deepth'
    height = 'height'
    width = 'width'
    search = 'search'


class operator:
    class Utils:

        def is_graph(self, v):
            return isinstance(v, var_list)

        def as_graph(self, v):
            if isinstance(v, var_list):
                return v.graph
            elif hasattr(v, 'graph'):
                return v.graph
            else:
                return AssertionError('This is not a Tensor: %s' % (str(v)))

        def broadcast_attr(self, l, r):
            if operator.is_graph(l):
                lattr = l.attr
                lndim = l.ndim
            else:
                lattr = [None]
                lndim = 0
            if operator.is_graph(r):
                rattr = r.attr
                rndim = r.ndim
            else:
                rattr = [None]
                rndim = 0
            attr = []
            ndim_diff = lndim - rndim
            for i in range(abs(ndim_diff)):
                if ndim_diff > 0:
                    attr.append(lattr[i])
                else:
                    attr.append(rattr[i])
            for i in range(abs(ndim_diff), max(lndim, rndim)):
                if ndim_diff >= 0:
                    if rattr[i - ndim_diff] is not None:
                        attr.append(rattr[i - ndim_diff])
                    elif lattr[i] is not None:
                        attr.append(lattr[i])
                    else:
                        attr.append(None)
                else:

                    if rattr[i] is not None:
                        attr.append(rattr[i])
                    elif lattr[i - ndim_diff] is not None:
                        attr.append(lattr[i - ndim_diff])
                    else:
                        attr.append(None)
            return attr

        def reduce_attr(self, t, axis):
            if axis is None:
                return []
            if not isinstance(axis, (tuple, list)):
                true_axis = axis
                if axis < 0:
                    true_axis = t.ndim + axis
                attr = t.attr[:true_axis] + t.attr[true_axis + 1:]
            else:
                true_axis_list = []
                for ax in axis:
                    true_axis_list.append(ax if ax >= 0 else t.ndim + ax)
                axes = range(t.ndim)
                for tax in true_axis_list:
                    axes.remove(tax)
                attr = t.attr[axes]
            return attr

        def group_attr(self, tlist):
            attr = []
            for i in range(tlist[0].ndim):
                find = False
                for j in tlist:
                    if j.attr[i] is not None:
                        attr.append(j.attr[i])
                        find = True
                        break
                if not find:
                    attr.append(None)
            return attr

        def broadcast_size(self, l, r):
            if operator.is_graph(l):
                lsize = l.size
                lndim = l.ndim
            else:
                lsize = [None]
                lndim = 0
            if operator.is_graph(r):
                rsize = r.size
                rndim = r.ndim
            else:
                rsize = [None]
                rndim = 0
            size = []
            ndim_diff = lndim - rndim
            for i in range(abs(ndim_diff)):
                if ndim_diff > 0:
                    size.append(lsize[i])
                else:
                    size.append(rsize[i])
            for i in range(abs(ndim_diff), max(lndim, rndim)):
                if ndim_diff >= 0:
                    if rsize[i - ndim_diff] is not None:
                        size.append(rsize[i - ndim_diff])
                    elif lsize[i] is not None:
                        size.append(lsize[i])
                    else:
                        size.append(None)
                else:

                    if rsize[i] is not None:
                        size.append(rsize[i])
                    elif lsize[i - ndim_diff] is not None:
                        size.append(lsize[i - ndim_diff])
                    else:
                        size.append(None)
            return size

        def reduce_size(self, t, axis):
            if axis is None:
                return []
            if not isinstance(axis, (tuple, list)):
                true_axis = axis
                if axis < 0:
                    true_axis = t.ndim + axis
                size = t.size[:true_axis] + t.size[true_axis + 1:]
            else:
                true_axis_list = []
                for ax in axis:
                    true_axis_list.append(ax if ax >= 0 else t.ndim + ax)
                axes = range(t.ndim)
                for tax in true_axis_list:
                    axes.remove(tax)
                size = t.size[axes]
            return size

        def group_size(self, tlist):
            size = []
            for i in range(tlist[0].ndim):
                find = True
                for j in tlist:
                    if j.size[i] is None:
                        find = False
                        break
                if find:
                    size.append(sum([t.size[i] for t in tlist]))
            return size

        def preprocess_lr(self, l, r):
            l_ = l
            r_ = r
            if operator.is_graph(l):
                l_ = l.graph
            if operator.is_graph(r):
                r_ = r.graph
            attr = operator.utils.broadcast_attr(l, r)
            size = operator.utils.broadcast_size(l, r)
            return l_, r_, size, attr

        def hash(self, t):
            return t.graph.__hash__()

        def cast(self, t, dtype):
            o = Backend.operator.utils.cast(t.graph, dtype)
            return tensor(graph=o, size=t.size, attr=t.attr)

    class Matrix:

        def dot(self, l, r):
            size = l.size[:-1] + r.size[1:]
            attr = l.attr[:-1] + r.attr[1:]
            o = Backend.operator.matrix.dot(l.graph, r.graph)
            return tensor(graph=o, size=size, attr=attr)

        def transpose(self, t):
            o = Backend.operator.matrix.transpose(t.graph)
            return tensor(graph=o, size=o.size[::-1], attr=o.attr[::-1])

        def dimshuffle(self, t, order=None):
            size = []
            for i in order:
                if i in ['x', None]:
                    size.append(None)
                else:
                    size.append(t.size[i])
            attr = []
            for i in order:
                if i in ['x', None]:
                    attr.append(None)
                else:
                    attr.append(t.attr[i])
            o = Backend.operator.matrix.dimshuffle(t.graph, order)
            return tensor(graph=o, size=size, attr=attr)

        def tile(self, t, n, attr=None):
            size = t.size
            for i, s in enumerate(size):
                if s is None:
                    if isinstance(n, (list, tuple)):
                        if n[i] != 1:
                            size[i] = n[i]
                else:
                    if isinstance(n, (list, tuple)):
                        if n[i] != 1:
                            size[i] = size[i] * n[i]
            o = Backend.operator.tile(t, n)
            if attr is None:
                attr = t.attr
            return tensor(graph=o, size=size, attr=attr)

        def repeat(self, t, n, attr=None):
            size = t.size
            for i, s in enumerate(size):
                if s is None:
                    if isinstance(n, (list, tuple)):
                        if n[i] != 1:
                            size[i] = n[i]
                else:
                    if isinstance(n, (list, tuple)):
                        if n[i] != 1:
                            size[i] = size[i] * n[i]
            o = Backend.operator.repeat(t, n)
            if attr is None:
                attr = t.attr
            return tensor(graph=o, size=size, attr=attr)

    class Elemwise:

        def add(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.add(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def sub(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.sub(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def mul(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.mul(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def div(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.div(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def floordiv(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.floordiv(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def mod(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.mod(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def divmod(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o1, o2 = Backend.operator.elemwise.divmod(l, r)
            return (tensor(graph=o1, size=size, attr=attr), tensor(graph=o2, size=size, attr=attr))

        def pow(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.pow(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def eq(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.eq(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def neq(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.neq(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def lt(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.lt(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def gt(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.gt(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def ge(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.ge(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def le(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.le(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def and_(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.and_(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def or_(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.or_(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def invert(self, t):
            o = Backend.operator.elemwise.invert(t.graph)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def xor(self, l, r):
            l, r, size, attr = operator.preprocess_lr(l, r)
            o = Backend.operator.elemwise.xor(l, r)
            return tensor(graph=o, size=size, attr=attr)

        def neg(self, t):
            o = Backend.operator.elemwise.neg(t.graph)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def abs(self, t):
            o = Backend.operator.elemwise.abs(t.graph)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def tanh(self, t):
            o = Backend.operator.elemwise.tanh(t.graph)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def sigmoid(self, t):
            o = Backend.operator.elemwise.sigmoid(t.graph)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def softmax(self, t, keepdims=True):
            o = Backend.operator.elemwise.softmax(t.graph, keepdims)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def relu(self, t):
            o = Backend.operator.elemwise.relu(t.graph)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def log(self, t):
            o = Backend.operator.elemwise.log(t.graph)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def exp(self, t):
            o = Backend.operator.elemwise.exp(t.graph)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def sqr(self, t):
            o = Backend.operator.elemwise.sqr(t.graph)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def sqrt(self, t):
            o = Backend.operator.elemwise.sqrt(t.graph)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def round(self, t):
            o = Backend.operator.elemwise.round(t.graph)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def clip(self, t, min, max):
            o = Backend.operator.elemwise.clip(t.graph, min, max)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def switch(self, condition, t, f):
            attr = operator.broadcast_attr(t, f)
            o = Backend.operator.elemwise.switch(operator.as_graph(condition), operator.as_graph(t),
                                                 operator.as_graph(f))
            return tensor(graph=o, size=t.size, attr=attr)

    class Reduction:

        def sum(self, t, axis=None, keepdims=False):
            if keepdims:
                size = t.size
            else:
                size = operator.reduce_size(t, axis)
            if keepdims:
                attr = t.attr
            else:
                attr = operator.reduce_attr(t, axis)
            o = Backend.operator.reduction.sum(t.graph, axis, keepdims)
            return tensor(graph=o, size=size, attr=attr)

        def mean(self, t, axis=None, keepdims=False):
            if keepdims:
                size = t.size
            else:
                size = operator.reduce_size(t, axis)
            if keepdims:
                attr = t.attr
            else:
                attr = operator.reduce_attr(t, axis)
            o = Backend.operator.reduction.mean(t.graph, axis, keepdims)
            return tensor(graph=o, size=size, attr=attr)

        def var(self, t, axis=None, keepdims=False):
            if keepdims:
                size = t.size
            else:
                size = operator.reduce_size(t, axis)
            if keepdims:
                attr = t.attr
            else:
                attr = operator.reduce_attr(t, axis)
            o = Backend.operator.reduction.var(t.graph, axis, keepdims)
            return tensor(graph=o, size=size, attr=attr)

        def std(self, t, axis=None, keepdims=False):
            if keepdims:
                size = t.size
            else:
                size = operator.reduce_size(t, axis)
            if keepdims:
                attr = t.attr
            else:
                attr = operator.reduce_attr(t, axis)
            o = Backend.operator.reduction.std(t.graph, axis, keepdims)
            return tensor(graph=o, size=size, attr=attr)

        def max(self, t, axis=None, keepdims=False):
            if keepdims:
                size = t.size
            else:
                size = operator.reduce_size(t, axis)
            if keepdims:
                attr = t.attr
            else:
                attr = operator.reduce_attr(t, axis)
            o = Backend.operator.reduction.max(t.graph, axis, keepdims)
            return tensor(graph=o, size=size, attr=attr)

        def argmax(self, t, axis=None, keepdims=False):
            if keepdims:
                size = t.size
            else:
                size = operator.reduce_size(t, axis)
            if keepdims:
                attr = t.attr
            else:
                attr = operator.reduce_attr(t, axis)
            o = Backend.operator.reduction.argmax(t.graph, axis, keepdims)
            return tensor(graph=o, size=size, attr=attr)

        def nonzero(self, t):
            o = Backend.operator.reduction.nonzero(t)
            return tensor(graph=o, size=[], attr=[])

    class Slicing:

        def getitem(self, t, key):
            def check(t, k):
                n = 0
                has_none = False
                for i in k:
                    if i is not None:
                        n += 1
                    else:
                        has_none = True
                if has_none:
                    if n != t.ndim:
                        raise AssertionError('Dim not match!')
                else:
                    if n > t.ndim:
                        raise AssertionError('Dim not match!')

            lkey = list(key)
            if not isinstance(key, tuple):
                lkey = [key]
                check(t, (key,))
                o = Backend.operator.slicing.getitem(t.graph, operator.as_graph(key))
            else:
                check(t, key)
                gkey = tuple([operator.as_graph(k) for k in key])
                o = Backend.operator.slicing.getitem(t.graph, gkey)
            for i in range(len(lkey), t.ndim):
                lkey.append(slice(None, None, None))
            size = []
            attr = []
            n = 0
            for k in lkey:
                if k is not None:
                    if operator.is_graph(k):
                        if k.ndim > 1:
                            raise AssertionError('slice indice must be int or list')
                        elif k.ndim == 1:
                            size.append(k.size[0])
                            if t.attr[n] is not None:
                                attr.append(t.attr[n])
                            else:
                                attr.append(k.attr[0])
                    elif isinstance(k, slice):
                        if operator.is_graph(k.start) or operator.is_graph(k.stop) or operator.is_graph(k.step):
                            length = None
                        else:
                            length = (k.stop - k.start) // k.step
                        size.append(length)
                        attr.append(t.attr[n])
                    elif isinstance(k, (tuple, list)):
                        size.append(len(k))
                        attr.append(t.attr[n])
                    n += 1
                else:
                    size.append(None)
                    attr.append(None)
            return tensor(graph=o, size=size, attr=attr)

        def setitem(self, t, key, tnew):
            tnew = operator.as_graph(tnew)
            o = Backend.operator.slicing.setitem(t.graph, key, tnew)
            return tensor(graph=o, size=t.size, attr=t.attr)

    class Grouping:

        def concatenate(self, tlist, axis=0):
            glist = [t.graph for t in tlist]
            size = operator.group_size(tlist)
            attr = operator.group_attr(tlist)
            o = Backend.operator.grouping.concatenate(glist, axis)
            return tensor(graph=o, size=size, attr=attr)

        def stack(self, tlist, addition_role=None):
            glist = [t.graph for t in tlist]
            size = [len(tlist)] + operator.group_attr(tlist)
            attr = [addition_role] + operator.group_attr(tlist)
            o = Backend.operator.grouping.stack(glist)
            return tensor(graph=o, size=size, attr=attr)

        def flatten(self, t):
            o = Backend.operator.grouping.flatten(t.graph)
            attr = [t.attr[-1]]
            if None not in t.size:
                size = [np.prod(t.size)]
            else:
                size = [None]
            return tensor(graph=o, size=size, attr=attr)

        def reshape(self, t, shape, size, attr):
            shape = [operator.as_graph(s) for s in shape]
            o = Backend.operator.grouping.reshape(t.graph, shape)
            return tensor(graph=o, size=size, attr=attr)

    class Alloc:

        def ones(self, shape, attr, dtype=Backend.kernel.config.floatX):
            size = shape
            shape = [operator.as_graph(s) for s in shape]
            o = Backend.operator.alloc.ones(shape, dtype)
            return tensor(graph=o, size=size, attr=attr)

        def alloc(self, value, shape, attr, dtype=Backend.kernel.config.floatX):
            size = shape
            shape = [operator.as_graph(s) for s in shape]
            o = Backend.operator.alloc.alloc(value, shape, dtype)
            return tensor(graph=o, size=size, attr=attr)

        def oneslike(self, t, dtype=Backend.kernel.config.floatX):
            o = Backend.operator.alloc.oneslike(t.graph, dtype)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def zeros(self, shape, attr, dtype=Backend.kernel.config.floatX):
            size = shape
            shape = [operator.as_graph(s) for s in shape]
            o = Backend.operator.alloc.zeros(shape, dtype)
            return tensor(graph=o, size=size, attr=attr)

        def zeroslike(self, t, dtype=Backend.kernel.config.floatX):
            o = Backend.operator.alloc.zeroslike(t.graph, dtype)
            return tensor(graph=o, size=t.size, attr=t.attr)

        def arange(self, *range, **kwargs):
            dtype = Backend.kernel.config.floatX
            if 'dtype' in kwargs:
                dtype = kwargs['dtype']
            if len(range) == 1:
                start = 0
                end = range[0]
                step = 1
            elif len(range) == 1:
                start = range[0]
                end = range[1]
                step = 1
            else:
                start = range[0]
                end = range[1]
                step = range[2]
            if operator.is_graph(start) or operator.is_graph(end) or operator.is_graph(step):
                length = None
            else:
                length = (end - start) // step
            start = operator.as_graph(start)
            end = operator.as_graph(end)
            step = operator.as_graph(step)
            o = Backend.operator.alloc.arange(start, end, step, dtype)
            attr = [None] * o.ndim
            return tensor(graph=o, size=[length], attr=attr)

        def constant(self, x, size=None, attr=None, name=None, ndim=None, dtype=Backend.kernel.config.floatX):
            x = operator.as_graph(x)
            o = Backend.operator.alloc.constant(x, name, ndim, dtype)
            return tensor(graph=o, size=size, attr=attr, name=name)

    class Nnet:
        def slice(self, x, i, len=None):
            if len is None:
                len = x.shape[-1] // 2
            key = [slice(None, None, None)] * (x.ndim - 1) + [slice(len * i, len * (i + 1), None)]
            key = tuple(key)
            return x[key]

        def lookup(self, x, dictionary, dim=None):
            if dim is None:
                if dictionary.size[-1] is not None:
                    dim = dictionary.size[-1]
                else:
                    dim = dictionary.shape[-1]
            if x.ndim > 1:
                embbedding = operator.reshape(dictionary[x.flatten()], shape=x.shape + [dim], size=x.size + [dim],
                                              attr=x.attr + [dictionary.attr[-1]])
                return embbedding
            else:
                embbedding = dictionary[x]
                return embbedding

        def shiftright(self, x, distance=1, shape=None, axis=1):
            if shape is None:
                background = operator.zeroslike(x)
            else:
                background = operator.zeros(shape, x.attr)
            bkey = tuple([slice(None, None, None)] * (axis - 1) + [slice(distance, None, None)])
            xkey = tuple([slice(None, None, None)] * (axis - 1) + [slice(None, -distance, None)])
            return operator.setitem(background, bkey, x[xkey])

        def shiftleft(self, x, distance=1, shape=None, axis=1):
            if shape is None:
                background = operator.zeroslike(x)
            else:
                background = operator.zeros(shape, x.attr)
            bkey = tuple([slice(None, None, None)] * (axis - 1) + [slice(None, -distance, None)])
            xkey = tuple([slice(None, None, None)] * (axis - 1) + [slice(distance, None, None)])
            return operator.setitem(background, bkey, x[xkey])

        def tailflatten(self, x, left_ndim=2, size=None, attr=None):
            ndim = left_ndim - 1
            if size is None:
                if None not in (x.size[ndim:]):
                    size = x.size[:ndim] + np.prod(x.size[ndim:])
                else:
                    size = x.size[:ndim] + [None]
            if attr is None:
                if None not in (x.size[ndim:]):
                    rest_attr = None
                    for i in x.attr[ndim:][::-1]:
                        if i is not None:
                            rest_attr = i
                    attr = x.attr[:ndim] + [rest_attr]
                else:
                    attr = x.attr[:ndim] + [None]
            return x.reshape([x.shape[0], -1], size, attr)

        def headflatten(self, x, left_ndim=2, size=None, attr=None):
            ndim = left_ndim - 1
            if size is None:
                if None not in (x.size[:ndim]):
                    size = np.prod(x.size[:ndim]) + x.size[ndim:]
                else:
                    size = [None] + x.size[ndim:]
            if attr is None:
                if None not in (x.size[:ndim]):
                    rest_attr = None
                    for i in x.attr[:ndim][::-1]:
                        if i is not None:
                            rest_attr = i
                    attr = [rest_attr] + x.attr[ndim:]
                else:
                    attr = [None] + x.attr[ndim:]
            return x.reshape([x.shape[0], -1], size, attr)

        def forwardbroadcast(self, x, ndim):
            key = range(x.ndim) + ['x'] * ndim
            return x.dimshuffle(key)

        def forwardbroadcastitem(self, x, ndim):
            key = [slice(None, None, None)] * x.ndim + [None] * ndim
            key = tuple(key)
            return x[key]

        def maxout(self, x, num_pieces):
            last_dim = x.shape[-1]
            output_dim = last_dim // num_pieces
            new_shape = ([x.shape[i] for i in range(x.ndim - 1)] +
                         [output_dim, num_pieces])
            return operator.max(operator.reshape(x, new_shape, x.size + [num_pieces], x.attr + [None]), axis=x.ndim)

        def glu(self, x, dim=None):
            x0 = operator.slice(x, 0, dim)
            x1 = operator.slice(x, 1, dim)
            return operator.sigmoid(x0) * x1

        def gtu(self, x, dim=None):
            x0 = operator.slice(x, 0, dim)
            x1 = operator.slice(x, 1, dim)
            return operator.sigmoid(x0) * operator.tanh(x1)

        def conv(self, input, filters, mode='normal', pad=None, stride=None,
                 dilation=None):
            '''

            :param input:
            :param filters:
            :param input_shape:
            :param filter_shape:[nfilter,nchannel,]
            :param mode: 'normal','full','half','pad'
            :param pad: default None or tuple()
            :param stride:
            :param dilation:default None or tuple()
            :param flip:
            :return:
            '''
            input = operator.utils.as_graph(input)
            filters = operator.utils.as_graph(filters)
            if stride is None:
                stride = tuple([1] * (input.ndim - 2))
            if not isinstance(stride, (tuple, list)):
                stride = [stride] * (input.ndim - 2)
            if dilation is None:
                dilation = tuple([1] * (input.ndim - 2))
            if not isinstance(dilation, (tuple, list)):
                dilation = [dilation] * (input.ndim - 2)
            if not isinstance(pad, (tuple, list)):
                if pad is not None:
                    pad = [pad] * (input.ndim - 2)
            o = Backend.operator.matrix.conv(input=input, filters=filters, input_shape=input.size,
                                             filter_shape=filters.size,
                                             mode=mode, pad=pad, stride=stride, dilation=dilation, flip=True)
            outchannel = filters.size[0]
            if pad is None:
                pad = tuple([0] * (input.ndim - 2))
            if mode is 'normal':
                outimagesize = [(a - (f + 2 * (d - 1)) + p * 2) // s + 1 for a, f, s, p, d in
                                zip(input.size[2:], filters.size[2:], stride, pad, dilation)]
            elif mode is 'full':
                outimagesize = [(a + (f + 2 * (d - 1)) + p * 2) // s - 1 for a, f, s, p, d in
                                zip(input.size[2:], filters.size[2:], stride, pad, dilation)]
            elif mode is 'half':
                outimagesize = input.size[2:]
            else:
                outimagesize = input.size[2:]
            attr = input.attr
            size = [outchannel] + outimagesize
            return tensor(graph=o, size=size, attr=attr)

        def cross_corr(self, input, filters, input_shape, filter_shape, mode, pad, stride, dilation):
            '''

            :param input:
            :param filters:
            :param input_shape:
            :param filter_shape:
            :param mode:
            :param pad:
            :param stride:
            :param dilation:
            :return:
            '''

            input = operator.utils.as_graph(input)
            filters = operator.utils.as_graph(filters)
            if stride is None:
                stride = tuple([1] * (input.ndim - 2))
            if not isinstance(stride, (tuple, list)):
                stride = [stride] * (input.ndim - 2)
            if dilation is None:
                dilation = tuple([1] * (input.ndim - 2))
            if not isinstance(dilation, (tuple, list)):
                dilation = [dilation] * (input.ndim - 2)
            if not isinstance(pad, (tuple, list)):
                if pad is not None:
                    pad = [pad] * (input.ndim - 2)
            o = Backend.operator.matrix.conv(input=input, filters=filters, input_shape=input.size,
                                             filter_shape=filters.size,
                                             mode=mode, pad=pad, stride=stride, dilation=dilation, flip=False)
            outchannel = filters.size[0]
            if pad is None:
                pad = tuple([0] * (input.ndim - 2))
            if mode is 'normal':
                outimagesize = [(a - (f + 2 * (d - 1)) + p * 2) // s + 1 for a, f, s, p, d in
                                zip(input.size[2:], filters.size[2:], stride, pad, dilation)]
            elif mode is 'full':
                outimagesize = [(a + (f + 2 * (d - 1)) + p * 2) // s - 1 for a, f, s, p, d in
                                zip(input.size[2:], filters.size[2:], stride, pad, dilation)]
            elif mode is 'half':
                outimagesize = input.size[2:]
            else:
                outimagesize = input.size[2:]
            attr = input.attr
            size = input.size[0] + [outchannel] + outimagesize
            return tensor(graph=o, size=size, attr=attr)

        def pool(self, input, window, mode='max', stride=None, pad=None, autopad=False):
            '''

            :param input:
            :param window:
            :param autopad:
            :param stride:
            :param pad:
            :param mode: 'max','sum','avg','avgpad'
            :return:
            '''
            input = operator.utils.as_graph(input)
            window = operator.utils.as_graph(window)
            if stride is None:
                stride = window
            if not isinstance(stride, (tuple, list)):
                stride = [stride] * (input.ndim - 2)
            if not isinstance(pad, (tuple, list)):
                if pad is not None:
                    pad = [pad] * (input.ndim - 2)
            if pad is None:
                pad = tuple([0] * (input.ndim - 2))
            o = Backend.operator.matrix.pool(input=input, window=window, mode=mode, stride=stride, pad=pad,
                                             autopad=autopad)
            if pad is None:
                pad = tuple([0] * (input.ndim - 2))
            if autopad:
                outimagesize = [(a - w + p * 2) // s + 1 for a, w, s, p in zip(input.size[2:], window, stride, pad)]
            else:
                outimagesize = [(a - w + p * 2 - 1) // s + 2 for a, w, s, p in
                                zip(input.size[2:], window, stride, pad)]
            size = input.size[:2] + outimagesize
            attr = input.attr
            return tensor(graph=o, size=size, attr=attr)

        def im2col(self, tensor, shape, step=None, mode='normal'):
            '''

            :param tensor:
            :param shape:
            :param step:
            :param mode:
            :return:
            '''
            return None

        def col2im(self, tensor, shape, original_shape=None, mode='normal'):
            '''

            :param tensor:
            :param shape:
            :param original_shape:
            :param mode:
            :return:
            '''
            return None

        def loss_function(self, y, y_true, fn, shape=None, size=None, attr=None):
            if shape is None:
                shape = y_true.shape
            if attr is None:
                attr = y_true.attr
            if size is None:
                size = y_true.size
            if y_true.ndim != 1:
                true = y_true.flatten()
            else:
                true = y_true
            if y.ndim > 2:
                prob = operator.tailflatten(y, 2)
            else:
                prob = y
            prob_ = operator.as_graph(prob)
            true_ = operator.as_graph(true)
            total_loss = tensor(graph=fn(prob_, true_), size=true.size, attr=true.attr)
            if len(shape) > y_true.ndim:
                total_loss.reshape(shape, size, attr)
            return total_loss

        def binary_crossentropy(self, y, y_true):
            return operator.nnet.loss_function(y, y_true, Backend.operator.nnet.binary_crossentropy)

        def categorical_crossentropy(self, y, y_true):
            return operator.nnet.loss_function(y, y_true, Backend.operator.nnet.categorical_crossentropy)

    utils = Utils()
    matrix = Matrix()
    elemwise = Elemwise()
    reduction = Reduction()
    slicing = Slicing()
    grouping = Grouping()
    alloc = Alloc()
    nnet = Nnet()

    '''      -----------      '''
    ###       ShortCuts       ###
    '''      -----------      '''

    ### Utils ###
    hash = utils.hash
    cast = utils.cast
    is_graph = utils.is_graph
    as_graph = utils.as_graph
    preprocess_lr = utils.preprocess_lr
    group_size = utils.group_size
    reduce_size = utils.reduce_size
    broadcast_size = utils.broadcast_size
    group_attr = utils.group_attr
    reduce_attr = utils.reduce_attr
    broadcast_attr = utils.broadcast_attr

    ### Matrix ###
    dot = matrix.dot
    transpose = matrix.transpose
    dimshuffle = matrix.dimshuffle
    tile = matrix.tile
    repeat = matrix.repeat

    ### Elemwise ###
    # operator #
    add = elemwise.add
    sub = elemwise.sub
    mul = elemwise.mul
    div = elemwise.div
    floordiv = elemwise.floordiv
    mod = elemwise.mod
    divmod = elemwise.divmod
    pow = elemwise.pow
    neg = elemwise.neg
    abs = elemwise.abs
    log = elemwise.log
    exp = elemwise.exp
    sqr = elemwise.sqr
    sqrt = elemwise.sqrt
    round = elemwise.round
    clip = elemwise.clip
    # function #
    tanh = elemwise.tanh
    sigmoid = elemwise.sigmoid
    softmax = elemwise.softmax
    relu = elemwise.relu
    # logic #
    eq = elemwise.eq
    neq = elemwise.neq
    lt = elemwise.lt
    le = elemwise.le
    gt = elemwise.gt
    ge = elemwise.ge
    and_ = elemwise.and_
    or_ = elemwise.or_
    not_ = elemwise.invert
    xor = elemwise.xor
    switch = elemwise.switch

    ### Reduction ###
    sum = reduction.sum
    mean = reduction.mean
    var = reduction.var
    std = reduction.std
    max = reduction.max
    argmax = reduction.argmax
    nonzero = reduction.nonzero

    ### Grouping ###
    concatenate = grouping.concatenate
    stack = grouping.stack
    reshape = grouping.reshape
    flatten = grouping.flatten

    ### Slicing ###
    getitem = slicing.getitem
    setitem = slicing.setitem

    ### alloc ###
    constant = alloc.constant
    arange = alloc.arange
    ones = alloc.ones
    zeros = alloc.zeros
    oneslike = alloc.oneslike
    zeroslike = alloc.zeroslike

    ### nnet ###
    conv = nnet.conv
    cross_corr = nnet.cross_corr
    pool = nnet.pool
    im2col = nnet.im2col
    col2im = nnet.col2im
    categorical_crossentropy = nnet.categorical_crossentropy
    binary_crossentropy = nnet.binary_crossentropy
    slice = nnet.slice
    lookup = nnet.lookup
    glu = nnet.glu
    gtu = nnet.gtu
    maxout = nnet.maxout
    shiftleft = nnet.shiftleft
    shiftright = nnet.shiftright
    tailflatten = nnet.tailflatten
    headflatten = nnet.headflatten
    forwardbroadcast = nnet.forwardbroadcast
    forwardbroadcastitem = nnet.forwardbroadcastitem


class placeholder(Backend.placeholder):
    def __init__(self, name, size, attr, dtype=Backend.kernel.config.floatX):
        Backend.placeholder.__init__(self, name, size, attr, dtype)

    def set_test_value(self, value):
        self.test_value = value

    @property
    def shape(self):
        return [tensor(s, [], []) for s in Backend.placeholder.get_shape_graph(self)]

    def __hash__(self):
        return operator.hash(self)

    def __str__(self):
        return str(self.graph)

    def __float__(self):
        operator.cast(self, kernel.config.floatX)

    def __int__(self):
        operator.cast(self, kernel.config.intX)

    def __long__(self):
        operator.cast(self, kernel.config.catX)

    def __abs__(self):
        return operator.abs(self)

    def __neg__(self):
        return operator.neg(self)

    def __nonzero__(self):
        return operator.nonzero(self)

    def __add__(self, other):
        return operator.add(self, other)

    def __sub__(self, other):
        return operator.sub(self, other)

    def __mul__(self, other):
        return operator.mul(self, other)

    def __div__(self, other):
        return operator.div(self, other)

    def __floordiv__(self, other):
        return operator.floordiv(self, other)

    def __mod__(self, other):
        return operator.mod(self, other)

    def __pow__(self, power, modulo=None):
        return operator.pow(self, power)

    def __eq__(self, other):
        return operator.eq(self, other)

    def __gt__(self, other):
        return operator.gt(self, other)

    def __lt__(self, other):
        return operator.lt(self, other)

    def __ge__(self, other):
        return operator.ge(self, other)

    def __le__(self, other):
        return operator.le(self, other)

    def __and__(self, other):
        return operator.and_(self, other)

    def __or__(self, other):
        return operator.or_(self, other)

    def __invert__(self):
        return operator.not_(self)

    def __xor__(self, other):
        return operator.xor(self, other)

    def __divmod__(self, other):
        return operator.divmod(self, other)

    def __iadd__(self, other):
        return operator.add(self, other)

    def __isub__(self, other):
        return operator.sub(self, other)

    def __imul__(self, other):
        return operator.mul(self, other)

    def __idiv__(self, other):
        return operator.div(self, other)

    def __ifloordiv__(self, other):
        return operator.floordiv(self, other)

    def __imod__(self, other):
        return operator.mod(self, other)

    def __ipow__(self, power, modulo=None):
        return operator.pow(self, power)

    def __iand__(self, other):
        return operator.and_(self, other)

    def __ior__(self, other):
        return operator.or_(self, other)

    def __ixor__(self, other):
        return operator.xor(self, other)

    def __radd__(self, other):
        return operator.add(other, self)

    def __rsub__(self, other):
        return operator.sub(other, self)

    def __rmul__(self, other):
        return operator.mul(other, self)

    def __rdiv__(self, other):
        return operator.div(other, self)

    def __rfloordiv__(self, other):
        return operator.floordiv(other, self)

    def __rmod__(self, other):
        return operator.mod(other, self)

    def __rpow__(self, other):
        return operator.pow(other, self)

    def __rdivmod__(self, other):
        return operator.divmod(other, self)

    def __rand__(self, other):
        return operator.and_(other, self)

    def __ror__(self, other):
        return operator.or_(other, self)

    def __rxor__(self, other):
        return operator.xor(other, self)

    def __getitem__(self, item):
        return operator.getitem(self, item)

    def __setitem__(self, key, value):
        return operator.setitem(self, key, value)

    def sqr(self):
        return operator.sqr(self)

    def sqrt(self):
        return operator.sqrt(self)

    def sum(self, axis=None, keepdims=False):
        return operator.sum(self, axis, keepdims)

    def std(self, axis=None, keepdims=False):
        return operator.std(self, axis, keepdims)

    def var(self, axis=None, keepdims=False):
        return operator.var(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return operator.mean(self, axis, keepdims)

    def max(self, axis=None, keepdims=False):
        return operator.max(self, axis, keepdims)

    def argmax(self, axis=None, keepdims=False):
        return operator.argmax(self, axis, keepdims)

    @property
    def T(self):
        return operator.transpose(self)

    def flatten(self):
        return operator.flatten(self)

    def reshape(self, shape, size, attr):
        return operator.reshape(self, shape, size, attr)

    def dimshuffle(self, order):
        return operator.dimshuffle(self, order)

    def cast(self, dtype):
        return operator.cast(self, dtype)


class shared(Backend.shared):
    def __init__(self, value, name, attr):
        Backend.shared.__init__(self, value, name, attr)

    def set(self, value):
        Backend.shared.set(self, value)

    def get(self):
        return Backend.shared.get(self)

    @property
    def shape(self):
        return [tensor(s, [], []) for s in Backend.shared.get_shape_graph(self)]

    def __hash__(self):
        return operator.hash(self)

    def __str__(self):
        return str(self.graph)

    def __float__(self):
        operator.cast(self, kernel.config.floatX)

    def __int__(self):
        operator.cast(self, kernel.config.intX)

    def __long__(self):
        operator.cast(self, kernel.config.catX)

    def __abs__(self):
        return operator.abs(self)

    def __neg__(self):
        return operator.neg(self)

    def __nonzero__(self):
        return operator.nonzero(self)

    def __add__(self, other):
        return operator.add(self, other)

    def __sub__(self, other):
        return operator.sub(self, other)

    def __mul__(self, other):
        return operator.mul(self, other)

    def __div__(self, other):
        return operator.div(self, other)

    def __floordiv__(self, other):
        return operator.floordiv(self, other)

    def __mod__(self, other):
        return operator.mod(self, other)

    def __pow__(self, power, modulo=None):
        return operator.pow(self, power)

    def __eq__(self, other):
        return operator.eq(self, other)

    def __gt__(self, other):
        return operator.gt(self, other)

    def __lt__(self, other):
        return operator.lt(self, other)

    def __ge__(self, other):
        return operator.ge(self, other)

    def __le__(self, other):
        return operator.le(self, other)

    def __and__(self, other):
        return operator.and_(self, other)

    def __or__(self, other):
        return operator.or_(self, other)

    def __invert__(self):
        return operator.not_(self)

    def __xor__(self, other):
        return operator.xor(self, other)

    def __divmod__(self, other):
        return operator.divmod(self, other)

    def __iadd__(self, other):
        return operator.add(self, other)

    def __isub__(self, other):
        return operator.sub(self, other)

    def __imul__(self, other):
        return operator.mul(self, other)

    def __idiv__(self, other):
        return operator.div(self, other)

    def __ifloordiv__(self, other):
        return operator.floordiv(self, other)

    def __imod__(self, other):
        return operator.mod(self, other)

    def __ipow__(self, power, modulo=None):
        return operator.pow(self, power)

    def __iand__(self, other):
        return operator.and_(self, other)

    def __ior__(self, other):
        return operator.or_(self, other)

    def __ixor__(self, other):
        return operator.xor(self, other)

    def __radd__(self, other):
        return operator.add(other, self)

    def __rsub__(self, other):
        return operator.sub(other, self)

    def __rmul__(self, other):
        return operator.mul(other, self)

    def __rdiv__(self, other):
        return operator.div(other, self)

    def __rfloordiv__(self, other):
        return operator.floordiv(other, self)

    def __rmod__(self, other):
        return operator.mod(other, self)

    def __rpow__(self, other):
        return operator.pow(other, self)

    def __rdivmod__(self, other):
        return operator.divmod(other, self)

    def __rand__(self, other):
        return operator.and_(other, self)

    def __ror__(self, other):
        return operator.or_(other, self)

    def __rxor__(self, other):
        return operator.xor(other, self)

    def __getitem__(self, item):
        return operator.getitem(self, item)

    def __setitem__(self, key, value):
        return operator.setitem(self, key, value)

    def sqr(self):
        return operator.sqr(self)

    def sqrt(self):
        return operator.sqrt(self)

    def sum(self, axis=None, keepdims=False):
        return operator.sum(self, axis, keepdims)

    def std(self, axis=None, keepdims=False):
        return operator.std(self, axis, keepdims)

    def var(self, axis=None, keepdims=False):
        return operator.var(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return operator.mean(self, axis, keepdims)

    def max(self, axis=None, keepdims=False):
        return operator.max(self, axis, keepdims)

    def argmax(self, axis=None, keepdims=False):
        return operator.argmax(self, axis, keepdims)

    @property
    def T(self):
        return operator.transpose(self)

    def flatten(self):
        return operator.flatten(self)

    def reshape(self, shape, size, attr):
        return operator.reshape(self, shape, size, attr)

    def dimshuffle(self, order):
        return operator.dimshuffle(self, order)

    def cast(self, dtype):
        return operator.cast(self, dtype)


class tensor(Backend.tensor):
    def __init__(self, graph, size, attr, name=None):
        Backend.tensor.__init__(self, graph, size, attr, name)
        if len(size) != len(self.shape):
            raise AssertionError('shape and size not match!   shape:%s   size:%s' % (str(self.shape), str(size)))
        if len(attr) != len(self.shape):
            raise AssertionError('shape and attr not match!   shape:%s   attr:%s' % (str(self.shape), str(attr)))
        if self.ndim != self.graph.ndim:
            raise AssertionError('ndims not match!   tensor:%s   graph:%s' % (str(self.ndim), str(self.graph.ndim)))

    @property
    def shape(self):
        return [tensor(s, [], []) for s in Backend.tensor.get_shape_graph(self)]

    def __hash__(self):
        return operator.hash(self)

    def __str__(self):
        return str(self.graph)

    def __float__(self):
        operator.cast(self, kernel.config.floatX)

    def __int__(self):
        operator.cast(self, kernel.config.intX)

    def __long__(self):
        operator.cast(self, kernel.config.catX)

    def __abs__(self):
        return operator.abs(self)

    def __neg__(self):
        return operator.neg(self)

    def __nonzero__(self):
        return operator.nonzero(self)

    def __add__(self, other):
        return operator.add(self, other)

    def __sub__(self, other):
        return operator.sub(self, other)

    def __mul__(self, other):
        return operator.mul(self, other)

    def __div__(self, other):
        return operator.div(self, other)

    def __floordiv__(self, other):
        return operator.floordiv(self, other)

    def __mod__(self, other):
        return operator.mod(self, other)

    def __pow__(self, power, modulo=None):
        return operator.pow(self, power)

    def __eq__(self, other):
        return operator.eq(self, other)

    def __gt__(self, other):
        return operator.gt(self, other)

    def __lt__(self, other):
        return operator.lt(self, other)

    def __ge__(self, other):
        return operator.ge(self, other)

    def __le__(self, other):
        return operator.le(self, other)

    def __and__(self, other):
        return operator.and_(self, other)

    def __or__(self, other):
        return operator.or_(self, other)

    def __invert__(self):
        return operator.not_(self)

    def __xor__(self, other):
        return operator.xor(self, other)

    def __divmod__(self, other):
        return operator.divmod(self, other)

    def __iadd__(self, other):
        return operator.add(self, other)

    def __isub__(self, other):
        return operator.sub(self, other)

    def __imul__(self, other):
        return operator.mul(self, other)

    def __idiv__(self, other):
        return operator.div(self, other)

    def __ifloordiv__(self, other):
        return operator.floordiv(self, other)

    def __imod__(self, other):
        return operator.mod(self, other)

    def __ipow__(self, power, modulo=None):
        return operator.pow(self, power)

    def __iand__(self, other):
        return operator.and_(self, other)

    def __ior__(self, other):
        return operator.or_(self, other)

    def __ixor__(self, other):
        return operator.xor(self, other)

    def __radd__(self, other):
        return operator.add(other, self)

    def __rsub__(self, other):
        return operator.sub(other, self)

    def __rmul__(self, other):
        return operator.mul(other, self)

    def __rdiv__(self, other):
        return operator.div(other, self)

    def __rfloordiv__(self, other):
        return operator.floordiv(other, self)

    def __rmod__(self, other):
        return operator.mod(other, self)

    def __rpow__(self, other):
        return operator.pow(other, self)

    def __rdivmod__(self, other):
        return operator.divmod(other, self)

    def __rand__(self, other):
        return operator.and_(other, self)

    def __ror__(self, other):
        return operator.or_(other, self)

    def __rxor__(self, other):
        return operator.xor(other, self)

    def __getitem__(self, item):
        return operator.getitem(self, item)

    def __setitem__(self, key, value):
        return operator.setitem(self, key, value)

    def sqr(self):
        return operator.sqr(self)

    def sqrt(self):
        return operator.sqrt(self)

    def sum(self, axis=None, keepdims=False):
        return operator.sum(self, axis, keepdims)

    def std(self, axis=None, keepdims=False):
        return operator.std(self, axis, keepdims)

    def var(self, axis=None, keepdims=False):
        return operator.var(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        return operator.mean(self, axis, keepdims)

    def max(self, axis=None, keepdims=False):
        return operator.max(self, axis, keepdims)

    def argmax(self, axis=None, keepdims=False):
        return operator.argmax(self, axis, keepdims)

    @property
    def T(self):
        return operator.transpose(self)

    def flatten(self):
        return operator.flatten(self)

    def reshape(self, shape, size, attr):
        return operator.reshape(self, shape, size, attr)

    def dimshuffle(self, order):
        return operator.dimshuffle(self, order)

    def cast(self, dtype):
        return operator.cast(self, dtype)


class randomgraph:
    @staticmethod
    def binomial(shape=(), n=1, p=0.5, ndim=None, attr=None, dtype=Backend.kernel.config.floatX):
        size = shape
        shape = [operator.as_graph(s) for s in shape]
        if attr is None:
            attr = [None] * len(shape)
        if ndim is not None:
            if ndim > len(shape):
                size.extend([None] * (ndim - len(shape)))
                attr.extend([None] * (ndim - len(shape)))
        return tensor(Backend.randomgraph.binomial(shape, n, p, ndim, dtype), size=size, attr=attr)

    @staticmethod
    def uniform(shape=(), low=0.0, high=1.1, ndim=None, attr=None, dtype=Backend.kernel.config.floatX):
        size = shape
        shape = [operator.as_graph(s) for s in shape]
        if attr is None:
            attr = [None] * len(shape)
        if ndim is not None:
            if ndim > len(shape):
                size.extend([None] * (ndim - len(shape)))
                attr.extend([None] * (ndim - len(shape)))
        return tensor(Backend.randomgraph.uniform(shape, low, high, ndim, dtype=dtype), size=size, attr=attr)

    @staticmethod
    def normal(shape, avg, std, ndim=None, attr=None, dtype=Backend.kernel.config.floatX):
        size = shape
        shape = [operator.as_graph(s) for s in shape]
        if attr is None:
            attr = [None] * len(shape)
        if ndim is not None:
            if ndim > len(shape):
                size.extend([None] * (ndim - len(shape)))
                attr.extend([None] * (ndim - len(shape)))
        return tensor(Backend.randomgraph.normal(shape, avg, std, ndim, dtype=dtype), size=size, attr=attr)

    @staticmethod
    def random_integers(shape, low, high, ndim=None, attr=None, dtype=Backend.kernel.config.floatX):
        size = shape
        shape = [operator.as_graph(s) for s in shape]
        if attr is None:
            attr = [None] * len(shape)
        if ndim is not None:
            if ndim > len(shape):
                size.extend([None] * (ndim - len(shape)))
                attr.extend([None] * (ndim - len(shape)))
        return tensor(Backend.randomgraph.random_integers(shape, low, high, ndim, dtype=dtype), size=size, attr=attr)

    @staticmethod
    def choice(shape, a, p, ndim=None, attr=None, dtype=Backend.kernel.config.floatX):
        size = shape
        shape = [operator.as_graph(s) for s in shape]
        if attr is None:
            attr = [None] * len(shape)
        if ndim is not None:
            if ndim > len(shape):
                size.extend([None] * (ndim - len(shape)))
                attr.extend([None] * (ndim - len(shape)))
        return tensor(Backend.randomgraph.choice(shape, a, p, ndim, dtype=dtype), size=size, attr=attr)

    @staticmethod
    def poisson(shape, lam, ndim=None, attr=None, dtype=Backend.kernel.config.floatX):
        size = shape
        shape = [operator.as_graph(s) for s in shape]
        if attr is None:
            attr = [None] * len(shape)
        if ndim is not None:
            if ndim > len(shape):
                size.extend([None] * (ndim - len(shape)))
                attr.extend([None] * (ndim - len(shape)))
        return tensor(Backend.randomgraph.poisson(shape, lam, ndim, dtype=dtype), size=size, attr=attr)

    @staticmethod
    def permutation(shape, n, ndim=None, attr=None, dtype=Backend.kernel.config.floatX):
        size = shape
        shape = [operator.as_graph(s) for s in shape]
        if attr is None:
            attr = [None] * len(shape)
        if ndim is not None:
            if ndim > len(shape):
                size.extend([None] * (ndim - len(shape)))
                attr.extend([None] * (ndim - len(shape)))
        return tensor(Backend.randomgraph.permutation(shape, n, ndim, dtype=dtype), size=size, attr=attr)

    @staticmethod
    def shuffle_row_elements(input):
        input = operator.as_graph(input)
        return tensor(Backend.randomgraph.shuffle_row_elements(input), input.size, input.attr)

    @staticmethod
    def multinomial(shape=(), n=1, p=None, ndim=None, attr=None, dtype=Backend.kernel.config.catX):
        size = shape
        shape = [operator.as_graph(s) for s in shape]
        if attr is None:
            attr = [None] * len(shape)
        if ndim is not None:
            if ndim > len(shape):
                size.extend([None] * (ndim - len(shape)))
                attr.extend([None] * (ndim - len(shape)))
        return tensor(Backend.randomgraph.multinormial(shape, p, n, ndim, dtype), size=size, attr=attr)


class config:
    floatX = Backend.config.floatX
    intX = Backend.config.intX
    boolX = Backend.config.boolX
    catX = Backend.config.catX


class kernel:
    placeholder = placeholder
    tensor = tensor
    shared = shared
    operator = operator
    config = config
    backend = Backend.kernel.backend
    modules = Backend.kernel.modules
    random = Backend.kernel.random
    randomgraph = Backend.kernel.randomgraph

    @staticmethod
    def printing(graph, outfile=None):
        return Backend.kernel.printing(operator.as_graph(graph), outfile)

    @staticmethod
    def grad(y, w):
        y_ = y.graph
        if isinstance(w, list):
            w_ = [i.graph for i in w]
        else:
            w_ = w.graph
        g = Backend.kernel.grad(y_, w_)
        if isinstance(g, list):
            g_ = [tensor(j, i.size, i.attr) for i, j in zip(w, g)]
        else:
            g_ = tensor(g, w.size, w.attr)
        return g_

    @staticmethod
    def compile(inputs, outputs=None, updates=None, strict=True):
        if inputs is not None and inputs != []:
            inputs_ = [operator.as_graph(i) for i in inputs]
        else:
            inputs_ = []
        if outputs is not None and outputs != []:
            outputs_ = [operator.as_graph(i) for i in outputs]
        else:
            outputs_ = []
        if isinstance(updates, (dict, OrderedDict)):
            updates = updates.items()
        if updates is not None and updates != []:
            updates_ = [(operator.as_graph(i), operator.as_graph(j)) for i, j in updates]
        else:
            updates_ = []
        return Backend.kernel.compile(inputs=inputs_, outputs=outputs_, updates=updates_, strict=strict)

    @staticmethod
    def scan(fn, sequences=None, initiate_states=None, non_iters=None, non_sequences=None, n_steps=None,
             go_backwards=False):
        if sequences is None:
            sequences = []
        if initiate_states is None:
            initiate_states = []
        if non_iters is None:
            non_iters = []
        if non_sequences is None:
            non_sequences = []
        if n_steps is None:
            if sequences != [] and isinstance(sequences, list):
                n_steps = sequences[0].shape[0]
            else:
                n_steps = 0

        args_size_list = [i[0].size for i in sequences] + [i.size for i in initiate_states] + [i.size for i in
                                                                                               non_sequences]
        args_attr_list = [i[0].attr for i in sequences] + [i.attr for i in initiate_states] + [i.attr for i in
                                                                                               non_sequences]

        def step(*args):
            rargs = [tensor(arg, args_size_list[i], args_attr_list[i]) for i, arg in enumerate(args)]
            rs = fn(*rargs)
            if isinstance(rs, (list, tuple)):
                return [r.graph for r in rs]
            else:
                return rs.graph

        def fn_wrapper(fn, args):
            targs = []
            for i, arg in enumerate(args):
                targs.append(tensor(arg, args_size_list[i], args_attr_list[i]))
            outs = fn(*targs)
            out_size_list = [outs.size] if not isinstance(outs, (list, tuple)) else [o.size for o in outs]
            out_attr_list = [outs.attr] if not isinstance(outs, (list, tuple)) else [o.attr for o in outs]
            gouts = []
            for out in outs:
                gouts.append(out.graph)
            return out_size_list, out_attr_list, gouts

        sequences = [i.graph for i in sequences]
        outputs_info = [i.graph for i in initiate_states] + non_iters
        non_sequences = [i.graph for i in non_sequences]

        out_size_list, out_attr_list, o, u = Backend.kernel.scan(fn_wrapper=fn_wrapper, fn=step, sequences=sequences,
                                                                 outputs_info=outputs_info,
                                                                 non_sequences=non_sequences,
                                                                 n_steps=n_steps.graph, go_backwards=go_backwards)

        if operator.is_graph(n_steps):
            out_size_list = [n_steps.size + out_size for out_size in out_size_list]
            out_attr_list = [n_steps.attr + out_attr for out_attr in out_attr_list]
        else:
            out_size_list = [[n_steps] + out_size for out_size in out_size_list]
            out_attr_list = [[None] + out_attr for out_attr in out_attr_list]

        if isinstance(o, list):
            o_ = [tensor(graph=i, size=j, attr=k) for i, j, k in zip(o, out_size_list, out_attr_list)]
        else:
            o_ = [tensor(graph=o, size=out_size_list[0], attr=out_attr_list[0])]

        u_ = OrderedDict()
        for k, v in u.items():
            u_[tensor(k, [None] * k.ndim, [None] * k.ndim)] = tensor(v, [None] * v.ndim, [None] * v.ndim)

        return o_, u_


var_list = (placeholder, shared, tensor, Backend.placeholder, Backend.shared, Backend.tensor)
