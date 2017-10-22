# -*- coding: utf-8 -*-
"""
Created on  八月 03 23:44 2017

@author: aeloyq
"""


def defaultreturn():
    raise AssertionError('Not Implemented')


class operator(object):
    class Utils:
        def cast(self, t, dtype):
            defaultreturn()

    class Matrix:
        def dot(self, l, r):
            defaultreturn()

        def transpose(self, t):
            defaultreturn()

        def dimshuffle(self, t, order):
            defaultreturn()

        def conv(self, input, filters, input_shape, filter_shape, mode, pad, strides, flip, dilation):
            defaultreturn()

        def pool(self, input, window, noborder, strides, pad, mode):
            defaultreturn()

    class Elemwise:
        def add(self, l, r):
            defaultreturn()

        def sub(self, l, r):
            defaultreturn()

        def mul(self, l, r):
            defaultreturn()

        def div(self, l, r):
            defaultreturn()

        def floordiv(self, l, r):
            defaultreturn()

        def mod(self, l, r):
            defaultreturn()

        def divmod(self, l, r):
            defaultreturn()

        def pow(self, l, r):
            defaultreturn()

        def neg(self, t):
            defaultreturn()

        def abs(self, t):
            defaultreturn()

        def tanh(self, t):
            defaultreturn()

        def sigmoid(self, t):
            defaultreturn()

        def softmax(self, t, keepdims):
            defaultreturn()

        def relu(self, t):
            defaultreturn()

        def log(self, t):
            defaultreturn()

        def exp(self, t):
            defaultreturn()

        def sqr(self, t):
            defaultreturn()

        def sqrt(self, t):
            defaultreturn()

        def round(self, t):
            defaultreturn()

        def clip(self, t, min, max):
            defaultreturn()

        def eq(self, l, r):
            defaultreturn()

        def neq(self, l, r):
            defaultreturn()

        def lt(self, l, r):
            defaultreturn()

        def gt(self, l, r):
            defaultreturn()

        def ge(self, l, r):
            defaultreturn()

        def le(self, l, r):
            defaultreturn()

        def and_(self, l, r):
            defaultreturn()

        def or_(self, l, r):
            defaultreturn()

        def invert(self, t):
            defaultreturn()

        def xor(self, l, r):
            defaultreturn()

        def switch(self, condition, t, f):
            defaultreturn()

    class Reduction:
        def sum(self, t, axis, keepdims):
            defaultreturn()

        def mean(self, t, axis, keepdims):
            defaultreturn()

        def var(self, t, axis, keepdims):
            defaultreturn()

        def std(self, t, axis, keepdims):
            defaultreturn()

        def max(self, t, axis, keepdims):
            defaultreturn()

        def argmax(self, t, axis, keepdims):
            defaultreturn()

        def nonzero(self, t, keepdims):
            defaultreturn()

    class Slicing:
        def getitem(self, t, key):
            defaultreturn()

        def setitem(self, t, key, tnew):
            defaultreturn()

    class Grouping:
        def flatten(self, t):
            defaultreturn()

        def reshape(self, t, shape):
            defaultreturn()

        def concatenate(self, tlist, axis):
            defaultreturn()

        def stack(self, tlist):
            defaultreturn()

    class Alloc:
        def arange(self, start, stop, step, dtype):
            defaultreturn()

        def constant(self, x, name, ndim, dtype):
            defaultreturn()

        def ones(self, shape, dtype):
            defaultreturn()

        def oneslike(self, t, dtype):
            defaultreturn()

        def zeros(self, shape, dtype):
            defaultreturn()

        def zeroslike(self, t, dtype):
            defaultreturn()

        def alloc(self, value, shape, dtype):
            defaultreturn()

    class Nnet:
        def binary_crossentropy(self, y, y_true):
            defaultreturn()

        def categorical_crossentropy(self, y, y_true):
            defaultreturn()

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


class Randomgraph(object):
    def binomial(self, shape, n, p, ndim, dtype):
        defaultreturn()

    def uniform(self, shape, low, high, ndim, dtype):
        defaultreturn()

    def normal(self, shape, avg, std, ndim, dtype):
        defaultreturn()

    def random_integers(self, shape, low, high, ndim, dtype):
        defaultreturn()

    def choice(self, shape, a, p, ndim, dtype):
        defaultreturn()

    def poisson(self, shape, lam, ndim, dtype):
        defaultreturn()

    def permutation(self, shape, n, ndim, dtype):
        defaultreturn()

    def shuffle_row_elements(self, input):
        defaultreturn()

    def multinormial(self, shape, n, p, ndim, dtype):
        defaultreturn()


randomgraph = Randomgraph()


class Kernel(object):
    def change_random_seed(self, seed):
        defaultreturn()

    def change_randomgraph_seed(self, seed):
        defaultreturn()

    def printing(self, graph, outfile):
        defaultreturn()

    def grad(self, y, w):
        defaultreturn()

    def compile(self, inputs, outputs, updates, strict):
        defaultreturn()

    def scan(self, fn, sequences=None, outputs_info=None, non_sequences=None, n_steps=None, go_backwards=False):
        defaultreturn()


kernel = Kernel()
