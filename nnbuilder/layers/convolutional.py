# -*- coding: utf-8 -*-
"""
Created on  四月 26 21:08 2017

@author: aeloyq
"""

import numpy as np
from utils import *
from basic import *
from simple import *
from roles import *
from ops import *
from nnbuilder.kernel import *


class convolution(component):
    def __init__(self, **kwargs):
        component.__init__(self, **kwargs)

    def init_conv_params(self, filtersize, biased=False, dim=2, weight_name='Convw',
                         weight_role=convw, unit_name=''):
        if not isinstance(dim, (list, tuple)):
            poolsize = [2] * dim
        else:
            poolsize = dim
        self.allocate(param_init_functions.convweight, unit_name + weight_name, weight_role,
                      filtersize, poolsize=poolsize)
        if biased:
            self.allocate(param_init_functions.zeros, unit_name + 'Bi', bias, [filtersize[0]])

    def layer_conv(self, name, X, F='Convw', mode='normal', strides=None, pad=None, flip=True, dilation=None):
        return T.conv(X, self.params[name + F], filter_shape=self.shapes[name + F], mode=mode, strides=strides, pad=pad,
                      flip=flip, dilation=dilation)

    def apply_bias(self, name, tvar, biased):
        if biased:
            return tvar + self.params[name + 'Bi'].dimshuffle(['x', 0] + ['x'] * (tvar.ndim - 2))
        else:
            return tvar


class pooling(component):
    def __init__(self, **kwargs):
        component.__init__(self, **kwargs)

    def init_pool_params(self, n_filters, biased=False, unit_name=''):
        if biased:
            self.allocate(param_init_functions.zeros, unit_name + 'Bi', bias, [n_filters])

    def layer_pool(self, X, window, noborder=True, strides=None, pad=None, mode='max'):
        return T.pool(X, window, noborder=noborder, strides=strides, pad=pad, mode=mode)

    def apply_bias(self, name, tvar, biased):
        if biased:
            return tvar + self.params[name + 'Bi'].dimshuffle(['x', 0] + ['x'] * (tvar.ndim - 2))
        else:
            return tvar


class subsampling(pooling):
    def __init__(self, **kwargs):
        pooling.__init__(self, **kwargs)

    def init_subsample_params(self, nchannels, biased=False, weight_name='Ssw', weight_role=samplew, unit_name=''):
        self.allocate(param_init_functions.uniform, unit_name + weight_name, weight_role, [nchannels])
        self.init_pool_params(nchannels, biased, unit_name)

    def layer_subsampling(self, name, X, windowsize, W='Ssw', noborder=True, strides=None, pad=None, mode='max'):
        o = T.pool(X, windowsize, noborder=noborder, strides=strides, pad=pad, mode=mode)
        o = o * self.params[name + W].dimshuffle(['x', 0] + ['x'] * (X.ndim - 2))
        return o

    def apply_activation(self, tvar, activation):
        if activation is not None:
            return activation(tvar)
        else:
            return tvar


class flatting(component):
    def __init__(self, **kwargs):
        component.__init__(self, **kwargs)

    def layer_flatting(self, X):
        return X.reshape([X.shape[0], -1], [batch, unit])


class fwdconv(forward, convolution):
    units_name = 'Convolution'

    def __init__(self, nfilters, filtersize, biased=False, mode='normal', strides=None, pad=None,
                 dilation=None, flip=False, **kwargs):
        forward.__init__(self, [nfilters] + list(filtersize), **kwargs)
        convolution.__init__(self, **kwargs)
        self.units = [fwdconv.units_name]
        self.nchannels = None
        self.imagesize = None
        self.filtersize = list(filtersize)
        self.nfilters = nfilters
        self.biased = biased
        self.mode = mode
        self.strides = strides
        self.pad = pad
        self.dilation = dilation
        self.flip = flip

    def get_units(self):
        self.units_dim = {fwdconv.units_name: tuple([self.nfilters, self.nchannels] + self.filtersize)}

    def init_params(self):
        self.init_conv_params(self.units_dim.values()[0], self.biased, dim=len(self.imagesize),
                              unit_name=self.units[0])

    def apply(self, X):
        o = self.layer_conv(self.units[0], X, mode=self.mode,
                            strides=self.strides, pad=self.pad, flip=self.flip, dilation=self.dilation)
        o = self.apply_bias(self.units[0], o, self.biased)
        return o

    def set_in_dim(self, dim):
        forward.set_in_dim(self, dim)
        self.nchannels = dim[0]
        self.imagesize = dim[1:]
        self.get_units()

    def get_out_dim(self):
        outchannel = self.nfilters
        if self.mode is 'normal':
            outimagesize = [i - j + 1 for i, j in zip(self.imagesize, self.filtersize)]
        elif self.mode is 'full':
            outimagesize = [i + j - 1 for i, j in zip(self.imagesize, self.filtersize)]
        elif self.mode is 'half':
            outimagesize = self.imagesize.copy()
        else:
            outimagesize = self.imagesize.copy()
        self.out_dim = [outchannel] + outimagesize


class fwdpool(forward, pooling):
    units_name = 'Pooling'

    def __init__(self, windowsize=(2, 2), biased=True, noborder=True, strides=None, pad=None, mode='max', **kwargs):
        forward.__init__(self, list(windowsize), **kwargs)
        pooling.__init__(self, **kwargs)
        self.units = [fwdpool.units_name]
        self.units_dim = OrderedDict()
        self.windowsize = list(windowsize)
        self.biased = biased
        self.noborder = noborder
        self.strides = strides
        self.mode = mode
        self.pad = pad

    def init_params(self):
        self.init_pool_params(self.nchannels, self.biased, unit_name=self.units[0])

    def get_units(self):
        self.units_dim[fwdpool.units_name] = self.nchannels

    def apply(self, X):
        o = self.layer_pool(X, self.windowsize, noborder=self.noborder, strides=self.strides, pad=self.pad,
                            mode=self.mode)
        o = self.apply_bias(self.units[0], o, self.biased)
        return o

    def set_in_dim(self, dim):
        forward.set_in_dim(self, dim)
        self.nchannels = dim[0]
        self.imagesize = dim[1:]
        self.get_units()

    def get_out_dim(self):
        if self.noborder:
            outimagesize = [i // j for i, j in zip(self.imagesize, self.windowsize)]
        else:
            outimagesize = [(i - 1) // j + 1 for i, j in zip(self.imagesize, self.windowsize)]
        self.out_dim = [self.nchannels] + outimagesize


class fwdsubsample(fwdpool, subsampling):
    units_name = 'Subsample'

    def __init__(self, windowsize=(2, 2), biased=True, activation=T.sigmoid, noborder=True, strides=None, pad=None,
                 mode='max', **kwargs):
        forward.__init__(self, list(windowsize), **kwargs)
        subsampling.__init__(self, **kwargs)
        self.units = [fwdsubsample.units_name]
        self.units_dim = OrderedDict()
        self.windowsize = list(windowsize)
        self.biased = biased
        self.activation = activation
        self.noborder = noborder
        self.strides = strides
        self.mode = mode
        self.pad = pad

    def get_units(self):
        self.units_dim[fwdsubsample.units_name] = self.nchannels

    def init_params(self):
        self.init_subsample_params(self.nchannels, self.biased, unit_name=self.units[0])

    def apply(self, X):
        o = self.layer_subsampling(self.units[0], X, windowsize=self.windowsize, noborder=self.noborder,
                                   strides=self.strides, pad=self.pad, mode=self.mode)
        o = self.apply_bias(self.units[0], o, self.biased)
        o = self.apply_activation(o, self.activation)
        return o


class itgconv(integrate, convolution):
    units_name = 'AConvolution'

    def __init__(self, filters, filtersize, biased=False, mode='normal', strides=None, pad=None,
                 dilation=None, flip=False, **kwargs):
        integrate.__init__(self, **kwargs)
        convolution.__init__(self, **kwargs)
        self.units = []
        self.units_dim = OrderedDict()
        self.nchannels = None
        self.imagesize = None
        self.filtersize = list(filtersize)
        self.filters = list(filters)
        self.biased = biased
        self.mode = mode
        self.strides = strides
        self.pad = pad
        self.dilation = dilation
        self.flip = flip

    def get_units(self):
        for filter in self.filters:
            for i in range(filter[2]):
                n = itgconv.units_name + '_' + str(filter[0]) + '_' + str(filter[1]) + '_' + str(i)
                self.units_dim[n] = [1, filter[0]*(self.nchannels/sum([filter[0],filter[1]]))] + self.filtersize
        self.units = self.units_dim.keys()

    def init_params(self):
        for name, dim in self.units_dim.items():
            self.init_conv_params(dim, self.biased, dim=len(self.imagesize), unit_name=name)

    def apply(self, X):
        outs = []
        circle_shift = lambda choice, n: [i + 1 if i + 1 < n else 0 for i in choice]
        for filter in self.filters:
            name = itgconv.units_name + '_' + str(filter[0]) + '_' + str(filter[1])
            choice = []
            for i in range(self.nchannels // sum([filter[0], filter[1]])):
                for j in range(filter[0]):
                    choice.append(i * sum([filter[0], filter[1]]) + j)
            for i in range(filter[2]):
                out = self.layer_conv(name + '_' + str(i), X[:, choice], mode=self.mode, strides=self.strides,
                                      pad=self.pad,
                                      flip=self.flip, dilation=self.dilation)
                outs.append(out)
                choice = circle_shift(choice, self.nchannels)
        return T.concatenate(outs, axis=1)

    def set_in_dim(self, dim):
        integrate.set_in_dim(self, dim)
        self.nchannels = dim[0]
        self.imagesize = dim[1:]
        filters = []
        nfilters = 0

        def errorinfo(f):
            raise AssertionError(
                'filter not match channel, filter:%s   n_channel:%s' % (str(f), str(self.nchannels)))

        for filter in self.filters:
            if self.nchannels % sum(filter) != 0:
                errorinfo(filter)
            fltr = [filter[0], filter[1]]
            if filter[0] == 0 or filter[1] == 0:
                if filter[0] != 6 and filter[1] != 6:
                    errorinfo(filter)
                else:
                    nfilters += 1
                    fltr.append(1)
            else:
                fltr.append(sum(filter))
                nfilters += sum(filter)
            filters.append(fltr)
        self.filters = filters
        self.nfilters = nfilters
        self.get_units()

    def get_out_dim(self):
        outchannel = self.nfilters
        if self.mode is 'normal':
            outimagesize = [i - j + 1 for i, j in zip(self.imagesize, self.filtersize)]
        elif self.mode is 'full':
            outimagesize = [i + j - 1 for i, j in zip(self.imagesize, self.filtersize)]
        elif self.mode is 'half':
            outimagesize = self.imagesize.copy()
        else:
            outimagesize = self.imagesize.copy()
        self.out_dim = [outchannel] + outimagesize


class trfflat(transfer, flatting):
    units_name = 'Flating'

    def __init__(self, **kwargs):
        transfer.__init__(self, **kwargs)
        flatting.__init__(self, **kwargs)

    def get_out_dim(self):
        self.out_dim = np.prod(self.in_dim)

    def apply(self, X):
        return self.layer_flatting(X)


class convhidden(fwdconv, hidden):
    pass


class poolhidden(fwdpool, hidden):
    pass


class subsamplehidden(fwdsubsample, hidden):
    pass


class flattenhidden(trfflat, hidden):
    pass


class asymconvhidden(itgconv, hidden):
    pass


class conv(convhidden, entity):
    pass


class pool(poolhidden, entity):
    pass


class subsample(subsamplehidden, entity):
    pass


class flatten(flattenhidden, entity):
    pass


class asymconv(asymconvhidden, entity):
    pass
