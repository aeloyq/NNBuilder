# -*- coding: utf-8 -*-
"""
Created on  八月 03 23:42 2017


@author: aeloyq
"""

print("Using Backend Theano")

import theano
import theano.tensor as T
import theano.tensor.signal.pool as P
import numpy
import basic
from theano.tensor.shared_randomstreams import RandomStreams


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
    concat_size = sum([tt.shape[axis] for tt in tensor_list])

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


class operator(basic.operator):
    class Utils(basic.operator.Utils):
        def cast(self, t, dtype):
            return T.cast(t, dtype)

    class Matrix(basic.operator.Matrix):

        def dot(self, l, r):
            return T.dot(l, r)

        def transpose(self, t):
            return t.T

        def dimshuffle(self, t, order):
            od = []
            for i in order:
                if i is None:
                    od.append('x')
                else:
                    od.append(i)
            return t.dimshuffle(od)

        def conv(self, input, filters, input_shape, filter_shape, mode, pad, strides, flip, dilation):
            if mode == 'normal':
                border_mode = 'valid'
            elif mode == 'full':
                border_mode = 'full'
            elif mode == 'half':
                border_mode = 'half'
            elif mode == 'pad':
                border_mode = pad
            else:
                border_mode = 'normal'
            if input.ndim == 4:
                return T.nnet.conv2d(input=input, filters=filters, input_shape=input_shape, filter_shape=filter_shape,
                                     border_mode=border_mode, subsample=strides, filter_flip=flip,
                                     filter_dilation=dilation)
            elif input.ndim == 5:
                return T.nnet.conv3d(input=input, filters=filters, input_shape=input_shape, filter_shape=filter_shape,
                                     border_mode=border_mode, subsample=strides, filter_flip=flip,
                                     filter_dilation=dilation)
            else:
                basic.defaultreturn()

        def pool(self, input, window, noborder, strides, pad, mode):
            if mode == 'max':
                mode = 'max'
            elif mode == 'sum':
                mode = 'sum'
            elif mode == 'avg':
                mode = 'average_exc_pad'
            elif mode == 'avgpad':
                mode = 'average_inc_pad'
            else:
                mode = 'sum'
            if input.ndim == 4:
                return P.pool_2d(input=input, ws=window, ignore_border=noborder, stride=strides, pad=pad, mode=mode)
            elif input.ndim == 5:
                return P.pool_3d(input=input, ws=window, ignore_border=noborder, stride=strides, pad=pad, mode=mode)
            else:
                basic.defaultreturn()

    class Elemwise(basic.operator.Elemwise):

        def add(self, l, r):
            return l + r

        def sub(self, l, r):
            return l - r

        def mul(self, l, r):
            return l * r

        def div(self, l, r):
            return l / r

        def floordiv(self, l, r):
            return l // r

        def mod(self, l, r):
            return l % r

        def divmod(self, l, r):
            return T.divmod(l, r)

        def pow(self, l, r):
            return T.power(l, r)

        def neg(self, t):
            return -t

        def abs(self, t):
            return T.abs_(t)

        def tanh(self, t):
            return T.tanh(t)

        def sigmoid(self, t):
            return T.nnet.sigmoid(t)

        def softmax(self, t, keepdims):
            if t.ndim == 2:
                return T.nnet.softmax(t)
            else:
                trshp = t.reshape([-1, t.shape[-1]])
                orshp = T.nnet.softmax(trshp)
                if not keepdims:
                    return orshp
                else:
                    return orshp.reshape(t.shape)

        def relu(self, t):
            return T.nnet.relu(t)

        def log(self, t):
            return T.log(t)

        def exp(self, t):
            return T.exp(t)

        def sqr(self, t):
            return T.sqr(t)

        def sqrt(self, t):
            return T.sqrt(t)

        def round(self, t):
            return T.round(t)

        def clip(self, t, min, max):
            return T.clip(t, min, max)

        def eq(self, l, r):
            return T.eq(l, r)

        def neq(self, l, r):
            return T.neq(l, r)

        def lt(self, l, r):
            return T.lt(l, r)

        def gt(self, l, r):
            return T.gt(l, r)

        def ge(self, l, r):
            return T.ge(l, r)

        def le(self, l, r):
            return T.le(l, r)

        def and_(self, l, r):
            return T.and_(l, r)

        def or_(self, l, r):
            return T.or_(l, r)

        def xor(self, l, r):
            return T.xor(l, r)

        def invert(self, t):
            return T.invert(t)

        def switch(self, condition, t, f):
            return T.switch(condition, t, f)

    class Reduction(basic.operator.Reduction):

        def sum(self, t, axis, keepdims):
            return T.sum(t, axis, keepdims=keepdims)

        def mean(self, t, axis, keepdims):
            return T.mean(t, axis, keepdims=keepdims)

        def var(self, t, axis, keepdims):
            return T.var(t, axis, keepdims=keepdims)

        def std(self, t, axis, keepdims):
            return T.std(t, axis, keepdims=keepdims)

        def max(self, t, axis, keepdims):
            return T.max(t, axis, keepdims=keepdims)

        def argmax(self, t, axis, keepdims):
            return T.argmax(t, axis, keepdims=keepdims)

        def nonzero(self, t, keepdims):
            return T.nonzero(t, return_matrix=keepdims)

    class Slicing(basic.operator.Slicing):

        def getitem(self, t, key):
            return t[key]

        def setitem(self, t, key, tnew):
            return T.set_subtensor(t[key], tnew)

    class Grouping(basic.operator.Grouping):

        def flatten(self, t):
            return t.flatten()

        def reshape(self, t, shape):
            return t.reshape(shape)

        def concatenate(self, tlist, axis):
            return concatenate(tlist, axis)

        def stack(self, tlist):
            return T.stack(tlist)

    class Alloc(basic.operator.Alloc):

        def arange(self, start, stop, step, dtype):
            return T.arange(start, stop, step, dtype)

        def constant(self, x, name, ndim, dtype):
            return T.constant(x, name, ndim, dtype)

        def ones(self, shape, dtype):
            return T.ones(shape, dtype)

        def oneslike(self, t, dtype):
            return T.ones_like(t, dtype)

        def zeros(self, shape, dtype):
            return T.zeros(shape, dtype)

        def zeroslike(self, t, dtype):
            return T.zeros_like(t, dtype)

        def alloc(self, value, shape, dtype):
            return T.alloc(value, *tuple(shape), dtype=dtype)

    class Nnet(basic.operator.Nnet):

        def binary_crossentropy(self, y, y_true):
            return T.nnet.binary_crossentropy(y, y_true)

        def categorical_crossentropy(self, y, y_true):
            return T.nnet.categorical_crossentropy(y, y_true)

    '''      -----------      '''
    ###       ShortCuts       ###
    '''      -----------      '''

    ### Utils ###
    utils = Utils()
    cast = utils.cast

    ### Matrix ###
    matrix = Matrix()
    dot = matrix.dot
    transpose = matrix.transpose
    dimshuffle = matrix.dimshuffle
    conv = matrix.conv
    pool = matrix.pool

    ### Elemwise ###
    # operator #
    elemwise = Elemwise()
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
    reduction = Reduction()
    sum = reduction.sum
    mean = reduction.mean
    var = reduction.var
    std = reduction.std
    max = reduction.max
    argmax = reduction.argmax
    nonzero = reduction.nonzero

    ### Grouping ###
    grouping = Grouping()
    concatenate = grouping.concatenate
    stack = grouping.stack
    reshape = grouping.reshape
    flatten = grouping.flatten

    ### Slicing ###
    slicing = Slicing()
    getitem = slicing.getitem
    setitem = slicing.setitem

    ### alloc ###
    alloc = Alloc()
    constant = alloc.constant
    arange = alloc.arange
    ones = alloc.ones
    zeros = alloc.zeros
    oneslike = alloc.oneslike
    zeroslike = alloc.zeroslike

    ### nnet ###
    nnet = Nnet()
    categorical_crossentropy = nnet.categorical_crossentropy
    binary_crossentropy = nnet.binary_crossentropy


class placeholder:
    def __init__(self, name, attr, dtype):
        def get_placeholder(ndim, float=True, bit2=False, bit16=False, bit32=False, bit64=False):
            if float:
                if bit32:
                    if ndim == 1:
                        return T.fvector
                    if ndim == 2:
                        return T.fmatrix
                    if ndim == 3:
                        return T.ftensor3
                    if ndim == 4:
                        return T.ftensor4
                    if ndim == 5:
                        return T.ftensor5
                elif bit64:
                    if ndim == 1:
                        return T.dvector
                    if ndim == 2:
                        return T.dmatrix
                    if ndim == 3:
                        return T.dtensor3
                    if ndim == 4:
                        return T.dtensor4
                    if ndim == 5:
                        return T.dtensor5
            else:
                if bit2:
                    if ndim == 1:
                        return T.bvector
                    if ndim == 2:
                        return T.bmatrix
                    if ndim == 3:
                        return T.btensor3
                    if ndim == 4:
                        return T.btensor4
                    if ndim == 5:
                        return T.btensor5
                elif bit16:
                    if ndim == 1:
                        return T.wvector
                    if ndim == 2:
                        return T.wmatrix
                    if ndim == 3:
                        return T.wtensor3
                    if ndim == 4:
                        return T.wtensor4
                    if ndim == 5:
                        return T.wtensor5
                elif bit32:
                    if ndim == 1:
                        return T.ivector
                    if ndim == 2:
                        return T.imatrix
                    if ndim == 3:
                        return T.itensor3
                    if ndim == 4:
                        return T.itensor4
                    if ndim == 5:
                        return T.itensor5
                elif bit64:
                    if ndim == 1:
                        return T.lvector
                    if ndim == 2:
                        return T.lmatrix
                    if ndim == 3:
                        return T.ltensor3
                    if ndim == 4:
                        return T.ltensor4
                    if ndim == 5:
                        return T.ltensor5

        float, bit2, bit16, bit32, bit64 = False, False, False, False, False
        if dtype == 'float32':
            float = True
            bit32 = True
        if dtype == 'float64':
            float = True
            bit64 = True
        if dtype == 'int2':
            bit2 = True
        if dtype == 'int16':
            bit16 = True
        if dtype == 'int32':
            bit52 = True
        if dtype == 'int64':
            bit64 = True
        self.name = name
        self.attr = attr
        self.ndim = len(self.attr)
        self.graph = get_placeholder(self.ndim, float, bit2, bit16, bit32, bit64)(self.name)
        self.broadcastable = self.graph.broadcastable
        self.dtype = self.graph.dtype


class shared:
    def __init__(self, value, name, attr):
        self.name = name
        self.graph = theano.shared(value=value, name=name, borrow=True)
        self.attr = attr
        self.ndim = len(self.attr)
        self.broadcastable = self.graph.broadcastable
        self.dtype = self.graph.dtype

    def set(self, value):
        self.graph.set_value(value)

    def get(self):
        return self.graph.get_value()


class tensor:
    def __init__(self, graph, attr, name=None):
        self.name = name
        self.graph = graph
        self.attr = attr
        self.ndim = graph.ndim
        self.broadcastable = graph.broadcastable
        self.dtype = graph.dtype


class Randomgraph(basic.Randomgraph):
    def binomial(self, shape, n, p, ndim, dtype):
        return kernel.randomgraph.binomial(shape, n, p, ndim, dtype=dtype)

    def uniform(self, shape, low, high, ndim, dtype):
        return kernel.randomgraph.uniform(shape, low, high, ndim, dtype=dtype)

    def normal(self, shape, avg, std, ndim, dtype):
        return kernel.randomgraph.normal(shape, avg, std, ndim, dtype=dtype)

    def random_integers(self, shape, low, high, ndim, dtype):
        return kernel.randomgraph.random_integers(shape, low, high, ndim, dtype=dtype)

    def choice(self, shape, a, p, ndim, dtype):
        return kernel.randomgraph.choice(shape, a, False, p, ndim, dtype=dtype)

    def poisson(self, shape, lam, ndim, dtype):
        return kernel.randomgraph.poisson(shape, lam, ndim, dtype=dtype)

    def permutation(self, shape, n, ndim, dtype):
        return kernel.randomgraph.permutation(shape, n, ndim, dtype=dtype)

    def shuffle_row_elements(self, input):
        return kernel.randomgraph.shuffle_row_elements(input)

    def multinormial(self, shape, n, p, ndim, dtype):
        return kernel.randomgraph.multinomial(shape, n, p, ndim, dtype=dtype)


randomgraph = Randomgraph()


class config:
    floatX = theano.config.floatX
    boolX = 'int8'
    intX = 'int16'
    catX = 'int64'


class Kernel(basic.Kernel):
    placeholder = placeholder
    tensor = tensor
    shared = shared
    operator = operator
    config = config
    backend = theano
    modules = [theano, T]
    random = numpy.random.RandomState(1234)
    randomgraph = RandomStreams(1234)

    def printing(self, graph, outfile):
        return theano.printing.pydotprint(graph, outfile, scan_graphs=True, var_with_name_simple=True,
                                          print_output_file=False, format='png')

    def grad(self, y, w):
        return T.grad(y, w)

    def compile(self, inputs, outputs, updates, strict):
        if strict:
            return theano.function(inputs=inputs, outputs=outputs, updates=updates)
        else:
            return theano.function(inputs=inputs, outputs=outputs, updates=updates, on_unused_input='ignore')

    def scan(self, fn, sequences=None, outputs_info=None, non_sequences=None, n_steps=None, go_backwards=False):

        return theano.scan(fn=fn, sequences=sequences, outputs_info=outputs_info, non_sequences=non_sequences,
                           n_steps=n_steps, go_backwards=go_backwards)


kernel = Kernel()
