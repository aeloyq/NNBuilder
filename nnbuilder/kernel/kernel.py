# -*- coding: utf-8 -*-
"""
Created on  八月 03 23:44 2017

@author: aeloyq
"""
import os
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

bkd_knl = Backend.kernel
bkd_op = Backend.operator
bkd_phd = Backend.placeholder
bkd_srd = Backend.shared
bkd_tsr = Backend.tensor
bkd_rng = Backend.randomgraph
bkd_cfg = Backend.config


class operator:
    class Utils:

        def is_graph(self, v):
            return isinstance(v, var_list)

        def as_graph(self, v):
            return v.graph if isinstance(v, var_list) else v

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
                return [None]
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

        def preprocess_lr(self, l, r):
            l_ = l
            r_ = r
            if operator.is_graph(l):
                l_ = l.graph
            if operator.is_graph(r):
                r_ = r.graph
            attr = operator.utils.broadcast_attr(l, r)
            # if operator.is_graph(l) and operator.is_graph(r):
            #     attr = operator.utils.group_attr([l, r])
            # elif operator.is_graph(l):
            #     attr = l.attr
            # else:
            #     attr = r.attr
            return l_, r_, attr

        def hash(self, t):
            return t.graph.__hash__()

        def cast(self, t, dtype):
            o = bkd_op.utils.cast(t.graph, dtype)
            return tensor(graph=o, attr=t.attr)

    class Matrix:

        def dot(self, l, r):
            attr = l.attr[:-1] + r.attr[1:]
            o = bkd_op.matrix.dot(l.graph, r.graph)
            return tensor(graph=o, attr=attr)

        def transpose(self, t):
            o = bkd_op.matrix.transpose(t.graph)
            return tensor(graph=o, attr=o.attr[::-1])

        def dimshuffle(self, t, order=None):
            attr = []
            for i in order:
                if i in ['x', None]:
                    attr.append(None)
                else:
                    attr.append(t.attr[i])
            o = bkd_op.matrix.dimshuffle(t.graph, order)
            return tensor(graph=o, attr=attr)

        def conv(self, input, filters, input_shape=None, filter_shape=None, mode='normal', pad=None, strides=None,
                 flip=True, dilation=None):
            '''

            :param input:
            :param filters:
            :param input_shape:
            :param filter_shape:
            :param mode: 'normal','full','half','pad'
            :param pad: default None or tuple()
            :param strides:
            :param flip:
            :param dilation:default None or tuple()
            :return:
            '''
            attr = input.attr
            input = operator.utils.as_graph(input)
            filters = operator.utils.as_graph(filters)
            if strides is None:
                strides = tuple([1] * (input.ndim - 2))
            if dilation is None:
                dilation = tuple([1] * (input.ndim - 2))
            o = bkd_op.matrix.conv(input=input, filters=filters, input_shape=input_shape,
                                   filter_shape=filter_shape,
                                   mode=mode, pad=pad, strides=strides, flip=flip, dilation=dilation)
            return tensor(graph=o, attr=attr)

        def pool(self, input, window, noborder=True, strides=None, pad=None, mode='max'):
            '''

            :param input:
            :param window:
            :param noborder:
            :param strides:
            :param pad:
            :param mode: 'max','sum','avg','avgpad'
            :return:
            '''
            attr = input.attr
            input = operator.utils.as_graph(input)
            window = operator.utils.as_graph(window)
            if pad is None:
                pad = tuple([0] * (input.ndim - 2))
            o = bkd_op.matrix.pool(input=input, window=window, noborder=noborder, strides=strides, pad=pad,
                                   mode=mode)
            return tensor(graph=o, attr=attr)

    class Elemwise:

        def add(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.add(l, r)
            return tensor(graph=o, attr=attr)

        def sub(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.sub(l, r)
            return tensor(graph=o, attr=attr)

        def mul(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.mul(l, r)
            return tensor(graph=o, attr=attr)

        def div(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.div(l, r)
            return tensor(graph=o, attr=attr)

        def floordiv(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.floordiv(l, r)
            return tensor(graph=o, attr=attr)

        def mod(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.mod(l, r)
            return tensor(graph=o, attr=attr)

        def divmod(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o1, o2 = bkd_op.elemwise.divmod(l, r)
            return (tensor(graph=o1, attr=attr), tensor(graph=o2, attr=attr))

        def pow(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.pow(l, r)
            return tensor(graph=o, attr=attr)

        def eq(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.eq(l, r)
            return tensor(graph=o, attr=attr)

        def neq(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.neq(l, r)
            return tensor(graph=o, attr=attr)

        def lt(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.lt(l, r)
            return tensor(graph=o, attr=attr)

        def gt(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.gt(l, r)
            return tensor(graph=o, attr=attr)

        def ge(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.ge(l, r)
            return tensor(graph=o, attr=attr)

        def le(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.le(l, r)
            return tensor(graph=o, attr=attr)

        def and_(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.and_(l, r)
            return tensor(graph=o, attr=attr)

        def or_(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.or_(l, r)
            return tensor(graph=o, attr=attr)

        def invert(self, t):
            o = bkd_op.elemwise.invert(t.graph)
            return tensor(graph=o, attr=t.attr)

        def xor(self, l, r):
            l, r, attr = operator.preprocess_lr(l, r)
            o = bkd_op.elemwise.xor(l, r)
            return tensor(graph=o, attr=attr)

        def neg(self, t):
            o = bkd_op.elemwise.neg(t.graph)
            return tensor(graph=o, attr=t.attr)

        def abs(self, t):
            o = bkd_op.elemwise.abs(t.graph)
            return tensor(graph=o, attr=t.attr)

        def tanh(self, t):
            o = bkd_op.elemwise.tanh(t.graph)
            return tensor(graph=o, attr=t.attr)

        def sigmoid(self, t):
            o = bkd_op.elemwise.sigmoid(t.graph)
            return tensor(graph=o, attr=t.attr)

        def softmax(self, t, keepdims=True):
            o = bkd_op.elemwise.softmax(t.graph, keepdims)
            return tensor(graph=o, attr=t.attr)

        def relu(self, t):
            o = bkd_op.elemwise.relu(t.graph)
            return tensor(graph=o, attr=t.attr)

        def log(self, t):
            o = bkd_op.elemwise.log(t.graph)
            return tensor(graph=o, attr=t.attr)

        def exp(self, t):
            o = bkd_op.elemwise.exp(t.graph)
            return tensor(graph=o, attr=t.attr)

        def sqr(self, t):
            o = bkd_op.elemwise.sqr(t.graph)
            return tensor(graph=o, attr=t.attr)

        def sqrt(self, t):
            o = bkd_op.elemwise.sqrt(t.graph)
            return tensor(graph=o, attr=t.attr)

        def round(self, t):
            o = bkd_op.elemwise.round(t.graph)
            return tensor(graph=o, attr=t.attr)

        def clip(self, t, min, max):
            o = bkd_op.elemwise.clip(t.graph, min, max)
            return tensor(graph=o, attr=t.attr)

        def switch(self, condition, t, f):
            attr = operator.broadcast_attr(t, f)
            o = bkd_op.elemwise.switch(operator.as_graph(condition), operator.as_graph(t), operator.as_graph(f))
            return tensor(graph=o, attr=attr)

    class Reduction:

        def sum(self, t, axis=None, keepdims=False):
            if keepdims:
                attr = t.attr
            else:
                attr = operator.reduce_attr(t, axis)
            o = bkd_op.reduction.sum(t.graph, axis, keepdims)
            return tensor(graph=o, attr=attr)

        def mean(self, t, axis=None, keepdims=False):
            if keepdims:
                attr = t.attr
            else:
                attr = operator.reduce_attr(t, axis)
            o = bkd_op.reduction.mean(t.graph, axis, keepdims)
            return tensor(graph=o, attr=attr)

        def var(self, t, axis=None, keepdims=False):
            if keepdims:
                attr = t.attr
            else:
                attr = operator.reduce_attr(t, axis)
            o = bkd_op.reduction.var(t.graph, axis, keepdims)
            return tensor(graph=o, attr=attr)

        def std(self, t, axis=None, keepdims=False):
            if keepdims:
                attr = t.attr
            else:
                attr = operator.reduce_attr(t, axis)
            o = bkd_op.reduction.std(t.graph, axis, keepdims)
            return tensor(graph=o, attr=attr)

        def max(self, t, axis=None, keepdims=False):
            if keepdims:
                attr = t.attr
            else:
                attr = operator.reduce_attr(t, axis)
            o = bkd_op.reduction.max(t.graph, axis, keepdims)
            return tensor(graph=o, attr=attr)

        def argmax(self, t, axis=None, keepdims=False):
            if keepdims:
                attr = t.attr
            else:
                attr = operator.reduce_attr(t, axis)
            o = bkd_op.reduction.argmax(t.graph, axis, keepdims)
            return tensor(graph=o, attr=attr)

        def nonzero(self, t):
            o = bkd_op.reduction.nonzero(t)
            return tensor(graph=o, attr=[None])

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
                o = bkd_op.slicing.getitem(t.graph, operator.as_graph(key))
            else:
                check(t, key)
                gkey = tuple([operator.as_graph(k) for k in key])
                o = bkd_op.slicing.getitem(t.graph, gkey)
            attr = []
            for i in range(len(lkey), t.ndim):
                lkey.append(slice(None, None, None))
            n = 0
            for k in lkey:
                if k is not None:
                    if operator.is_graph(k):
                        if k.ndim != 0:
                            attr.append(t.attr[n])
                    elif isinstance(k, (slice, tuple, list)):
                        attr.append(t.attr[n])
                    n += 1
                else:
                    attr.append(None)
            if attr == []:
                attr = [None]
            return tensor(graph=o, attr=attr)

        def setitem(self, t, key, tnew):
            attr = t.attr
            tnew = operator.as_graph(tnew)
            o = bkd_op.slicing.setitem(t.graph, key, tnew)
            return tensor(graph=o, attr=attr)

    class Grouping:

        def concatenate(self, tlist, axis=0):
            glist = [t.graph for t in tlist]
            attr = operator.group_attr(tlist)
            o = bkd_op.grouping.concatenate(glist, axis)
            return tensor(graph=o, attr=attr)

        def stack(self, tlist, addition_role=None):
            glist = [t.graph for t in tlist]
            attr = [addition_role] + operator.group_attr(tlist)
            o = bkd_op.grouping.stack(glist)
            return tensor(graph=o, attr=attr)

        def flatten(self, t):
            o = bkd_op.grouping.flatten(t.graph)
            attr = [t.attr[-1]]
            return tensor(graph=o, attr=attr)

        def reshape(self, t, shape, attr):
            shape = [operator.as_graph(s) for s in shape]
            o = bkd_op.grouping.reshape(t.graph, shape)
            attr = attr
            return tensor(graph=o, attr=attr)

    class Alloc:

        def ones(self, shape, attr, dtype=bkd_knl.config.floatX):
            shape = [operator.as_graph(s) for s in shape]
            o = bkd_op.alloc.ones(shape, dtype)
            return tensor(graph=o, attr=attr)

        def alloc(self, value, shape, attr, dtype=bkd_knl.config.floatX):
            shape = [operator.as_graph(s) for s in shape]
            o = bkd_op.alloc.alloc(value, shape, dtype)
            return tensor(graph=o, attr=attr)

        def oneslike(self, t, dtype=bkd_knl.config.floatX):
            o = bkd_op.alloc.oneslike(t.graph, dtype)
            attr = t.attr
            return tensor(graph=o, attr=attr)

        def zeros(self, shape, attr, dtype=bkd_knl.config.floatX):
            shape = [operator.as_graph(s) for s in shape]
            o = bkd_op.alloc.zeros(shape, dtype)
            return tensor(graph=o, attr=attr)

        def zeroslike(self, t, dtype=bkd_knl.config.floatX):
            o = bkd_op.alloc.zeroslike(t.graph, dtype)
            attr = t.attr
            return tensor(graph=o, attr=attr)

        def arange(self, start, end=None, step=None, dtype=bkd_knl.config.floatX):
            start = operator.as_graph(start)
            end = operator.as_graph(end)
            step = operator.as_graph(step)
            o = bkd_op.alloc.arange(start, end, step, dtype)
            attr = [None] * o.ndim
            return tensor(graph=o, attr=attr)

        def constant(self, x, role=None, name=None, ndim=None, dtype=bkd_knl.config.floatX):
            x = operator.as_graph(x)
            o = bkd_op.alloc.constant(x, name, ndim, dtype)
            t = tensor(graph=o, attr=[role], name=name)

    class Nnet:
        def slice(self, x, i, len=None):
            if len is None:
                len = x.shape[-1] // 2
            key = [slice(None, None, None)] * (x.ndim - 1) + [slice(len * i, len * (i + 1), None)]
            key = tuple(key)
            return x[key]

        def lookup(self, x, dictionary, dim=None):
            if dim is None:
                dim = dictionary.shape[-1]
            if x.ndim > 1:
                embbedding = operator.reshape(dictionary[x.flatten()], shape=x.shape + [dim],
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

        def forwardflatten(self, x, ndim=2):
            return x.reshape([-1] + x.shape[:-(ndim - 1)], attr=[None] + x.attr[:-(ndim - 1)])

        def backwardflatten(self, x, ndim=2):
            return x.reshape(x.shape[(ndim - 1):] + [-1], attr=x.attr[(ndim - 1):] + [None])

        def forwardbroadcast(self, x, ndim):
            key = range(x.ndim) + [None] * ndim
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
            return operator.max(operator.reshape(x, new_shape, x.attr + [None]), axis=x.ndim)

        def glu(self, x, dim=None):
            x0 = operator.slice(x, 0, dim)
            x1 = operator.slice(x, 1, dim)
            return operator.sigmoid(x0) * x1

        def gtu(self, x, dim=None):
            x0 = operator.slice(x, 0, dim)
            x1 = operator.slice(x, 1, dim)
            return operator.sigmoid(x0) * operator.tanh(x1)

        def loss_function(self, y, y_true, fn, shape=None, attr=None):
            if shape is None:
                shape = y.shape[:-1]
            if attr is None:
                attr = y.attr[:-1]
            prob = y
            true = y_true
            if y_true.ndim != 1:
                true = y_true.flatten()
            if y.ndim > 2:
                prob = operator.forwardflatten(prob, 2)
            prob_ = operator.as_graph(prob)
            true_ = operator.as_graph(true)
            total_loss = tensor(graph=fn(prob_, true_), attr=[None])
            if len(shape) >= 2:
                total_loss.reshape(shape, attr)
                for _ in range(y.ndim - 1):
                    total_loss = total_loss.sum(0)
            loss = total_loss.mean()
            return loss

        def mean_square(self, y, y_true):
            def mse(y, y_true):
                return operator.elemwise.sqr(y[:, 0] - y_true)

            return operator.nnet.loss_function(y, y_true, mse)

        def root_mean_square(self, y, y_true):
            def rmse(y, y_true):
                return operator.elemwise.sqrt(operator.elemwise.sqr(y[:, 0] - y_true))

            return operator.nnet.loss_function(y, y_true, rmse)

        def binary_crossentropy(self, y, y_true):
            return operator.nnet.loss_function(y, y_true, bkd_op.nnet.binary_crossentropy)

        def categorical_crossentropy(self, y, y_true):
            return operator.nnet.loss_function(y, y_true, bkd_op.nnet.categorical_crossentropy)

        def softmax_categorical_crossentropy(self, output, y_true):
            y = operator.elemwise.softmax(output, keepdims=False)
            return operator.nnet.loss_function(y, y_true, bkd_op.nnet.categorical_crossentropy)

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
    group_attr = utils.group_attr
    reduce_attr = utils.reduce_attr
    broadcast_attr = utils.broadcast_attr

    ### Matrix ###
    dot = matrix.dot
    transpose = matrix.transpose
    dimshuffle = matrix.dimshuffle
    conv = matrix.conv
    pool = matrix.pool

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
    mean_square = nnet.mean_square
    root_mean_square = nnet.root_mean_square
    categorical_crossentropy = nnet.categorical_crossentropy
    binary_crossentropy = nnet.binary_crossentropy
    slice = nnet.slice
    lookup = nnet.lookup
    glu = nnet.glu
    gtu = nnet.gtu
    maxout = nnet.maxout
    shiftleft = nnet.shiftleft
    shiftright = nnet.shiftright
    forwardflatten = nnet.forwardflatten
    backwardflatten = nnet.backwardflatten
    forwardbroadcast = nnet.forwardbroadcast
    forwardbroadcastitem = nnet.forwardbroadcastitem


class placeholder(Backend.placeholder):
    def __init__(self, name, attr, dtype=bkd_knl.config.floatX):
        Backend.placeholder.__init__(self, name, attr, dtype)

    def set_test_value(self, value):
        self.test_value = value

    def type(self):
        return tensor(self.graph.type(), self.attr)

    @property
    def shape(self):
        return [tensor(s, [r]) for s, r in zip(self.graph.shape, self.attr)]

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

    def reshape(self, shape, attr):
        return operator.reshape(self, shape, attr)

    def dimshuffle(self, order):
        return operator.dimshuffle(self, order)

    def cast(self, dtype):
        return operator.cast(self, dtype)


class shared(Backend.shared):
    def __init__(self, value, name, attr):
        Backend.shared.__init__(self, value, name, attr)

    def set(self, value):
        bkd_srd.set(self, value)

    def get(self):
        return bkd_srd.get(self)

    def type(self):
        return tensor(self.graph.type(), self.attr)

    @property
    def shape(self):
        return [tensor(s, [r]) for s, r in zip(self.graph.shape, self.attr)]

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

    def reshape(self, shape, attr):
        return operator.reshape(self, shape, attr)

    def dimshuffle(self, order):
        return operator.dimshuffle(self, order)

    def cast(self, dtype):
        return operator.cast(self, dtype)


class tensor(Backend.tensor):
    def __init__(self, graph, attr, name=None):
        Backend.tensor.__init__(self, graph, attr, name)
        if len(attr) != len(self.shape) and len(self.shape) != 0:
            raise AssertionError('shape and attr not match!   shape:%s   attr:%s' % (str(self.shape), str(attr)))

        if self.ndim != self.graph.ndim:
            raise AssertionError('ndims not match!   tensor:%s   graph:%s' % (str(self.ndim), str(self.graph.ndim)))

    def type(self):
        return tensor(self.graph.type(), self.attr)

    @property
    def shape(self):
        return [tensor(s, [r]) for s, r in zip(self.graph.shape, self.attr)]

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

    def reshape(self, shape, attr):
        return operator.reshape(self, shape, attr)

    def dimshuffle(self, order):
        return operator.dimshuffle(self, order)

    def cast(self, dtype):
        return operator.cast(self, dtype)


class randomgraph:
    @staticmethod
    def binomial(shape=(), n=1, p=0.5, ndim=None, dtype=bkd_knl.config.floatX):
        shape = [operator.as_graph(s) for s in shape]
        return tensor(bkd_rng.binomial(shape, n, p, ndim, dtype), [None] * len(shape))

    @staticmethod
    def uniform(shape=(), low=0.0, high=1.1, ndim=None, dtype=bkd_knl.config.floatX):
        shape = [operator.as_graph(s) for s in shape]
        return tensor(bkd_rng.uniform(shape, low, high, ndim, dtype=dtype), [None] * len(shape))

    @staticmethod
    def normal(shape, avg, std, ndim=None, dtype=bkd_knl.config.floatX):
        shape = [operator.as_graph(s) for s in shape]
        return tensor(bkd_rng.normal(shape, avg, std, ndim, dtype=dtype), [None] * len(shape))

    @staticmethod
    def random_integers(shape, low, high, ndim=None, dtype=bkd_knl.config.floatX):
        shape = [operator.as_graph(s) for s in shape]
        return tensor(bkd_rng.random_integers(shape, low, high, ndim, dtype=dtype), [None] * len(shape))

    @staticmethod
    def choice(shape, a, p, ndim=None, dtype=bkd_knl.config.floatX):
        shape = [operator.as_graph(s) for s in shape]
        return tensor(bkd_rng.choice(shape, a, p, ndim, dtype=dtype), [None] * len(shape))

    @staticmethod
    def poisson(shape, lam, ndim=None, dtype=bkd_knl.config.floatX):
        shape = [operator.as_graph(s) for s in shape]
        return tensor(bkd_rng.poisson(shape, lam, ndim, dtype=dtype), [None] * len(shape))

    @staticmethod
    def permutation(shape, n, ndim=None, dtype=bkd_knl.config.floatX):
        shape = [operator.as_graph(s) for s in shape]
        return tensor(bkd_rng.permutation(shape, n, ndim, dtype=dtype), [None] * len(shape))

    @staticmethod
    def shuffle_row_elements(input):
        attr = input.attr
        input = operator.as_graph(input)
        return tensor(bkd_rng.shuffle_row_elements(input), attr)

    @staticmethod
    def multinomial(shape=(), n=1, p=None, ndim=None, dtype=bkd_knl.config.catX):
        shape = [operator.as_graph(s) for s in shape]
        return tensor(bkd_rng.multinormial(shape, p, n, ndim, dtype), [None] * len(shape))


class config:
    floatX = bkd_cfg.floatX
    intX = bkd_cfg.intX
    boolX = bkd_cfg.boolX
    catX = bkd_cfg.catX


class kernel:
    placeholder = placeholder
    tensor = tensor
    shared = shared
    operator = operator
    config = config
    backend = bkd_knl.backend
    modules = bkd_knl.modules
    random = bkd_knl.random
    randomgraph = bkd_knl.randomgraph

    @staticmethod
    def printing(graph, outfile=None):
        return bkd_knl.printing(operator.as_graph(graph), outfile)

    @staticmethod
    def grad(y, w):
        y_ = y.graph
        if isinstance(w, list):
            w_ = [i.graph for i in w]
        else:
            w_ = w.graph
        g = bkd_knl.grad(y_, w_)
        if isinstance(g, list):
            g_ = [tensor(j, i.attr) for i, j in zip(w, g)]
        else:
            g_ = tensor(g, w.attr)
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
        if updates is not None and updates != []:
            updates_ = [(operator.as_graph(i), operator.as_graph(j)) for i, j in updates]
        else:
            updates_ = []
        return bkd_knl.compile(inputs=inputs_, outputs=outputs_, updates=updates_, strict=strict)

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

        args_attr_list = [i[0].attr for i in sequences] + [i.attr for i in initiate_states] + [i.attr for i in
                                                                                               non_sequences]
        out = fn(*([seq[0] for seq in sequences] + initiate_states + non_sequences))
        out_attr_list = [out.attr] if not isinstance(out, (list, tuple)) else [o.attr for o in out]

        def step(*args):
            rargs = [tensor(arg, args_attr_list[i]) for i, arg in enumerate(args)]
            rs = fn(*rargs)
            if isinstance(rs, (list, tuple)):
                return [r.graph for r in rs]
            else:
                return rs.graph

        sequences = [i.graph for i in sequences]
        outputs_info = [i.graph for i in initiate_states] + non_iters
        non_sequences = [i.graph for i in non_sequences]

        if operator.is_graph(n_steps):
            out_attr_list = [n_steps.attr + out_attr for out_attr in out_attr_list]
        else:
            out_attr_list = [[None] + out_attr for out_attr in out_attr_list]

        o, u = bkd_knl.scan(fn=step, sequences=sequences, outputs_info=outputs_info, non_sequences=non_sequences,
                            n_steps=n_steps.graph, go_backwards=go_backwards)

        if isinstance(o, list):
            o_ = [tensor(graph=i, attr=j) for i, j in zip(o, out_attr_list)]
        else:
            o_ = [tensor(graph=o, attr=out_attr_list[0])]

        u_ = OrderedDict()
        for k, v in u.items():
            u_[tensor(k, [None])] = tensor(v, [None])

        return o_, u_


var_list = (placeholder, shared, tensor, bkd_phd, bkd_srd, bkd_tsr)
