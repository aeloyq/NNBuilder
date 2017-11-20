# -*- coding: utf-8 -*-
"""
Created on  四月 26 21:08 2017

@author: aeloyq
"""

from simple import *


class Conv(LayerBase):
    def __init__(self, nfilters, filtersize, bias=False, activation=None, stride=None, pad=None, dilation=None,
                 group=1, **kwargs):
        LayerBase.__init__(self, **kwargs)
        self.nfilters = nfilters
        self.filtersize = filtersize
        self.group = group
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.biased = bias
        self.activated = activation

    def set_children(self):
        if len(self.filtersize) == 2:
            if self.group == 1:
                self.filters = Parameter(self, 'Filters', Parameter.conv2w, random=Parameter.convweight)
            else:
                filters = OrderedDict()
                for i in range(self.group):
                    filters['Filters%d' % (i)] = Parameter(self, 'Filters', Parameter.conv2w,
                                                           random=Parameter.convweight)
                self.filtersdict = ParamDict(filters)
        else:
            if self.group == 1:
                self.filters = Parameter(self, 'Filters', Parameter.conv3w, random=Parameter.convweight)
            else:
                filters = OrderedDict()
                for i in range(self.group):
                    filters['Filters%d' % (i)] = Parameter(self, 'Filters', Parameter.conv3w,
                                                           random=Parameter.convweight)
                self.filtersdict = ParamDict(filters)
        if self.biased:
            self.bias = Bias()
        if self.activated is not None:
            self.activation = Activation(self.activation)
        self.nchannels = self.in_dim[0]
        if hasattr(self, 'scale'):
            self.scale.unit_dim = self.nchannels
        if hasattr(self, 'bias'):
            self.bias.unit_dim = self.nchannels

    def set_params(self):
        if self.group == 1:
            self.filters.shape = [self.nchannels] + self.filtersize
        else:
            for filter in self.filtersdict:
                filter.shape = [self.nchannels // self.group] + self.filtersize

    def apply(self, X):
        if self.group == 1:
            output = T.conv(X, self.filters(), pad=self.pad, stride=self.stride, dilation=self.dilation)
        else:
            outputs = []
            for i, filter in enumerate(self.filters):
                nchannels = self.nchannels // self.group
                outputs.append(
                    T.conv(X[:, nchannels * i:nchannels * (i + 1)], filter(),
                           pad=self.pad, stride=self.stride, dilation=self.dilation))
            output = T.concatenate(outputs, axis=1)
        if hasattr(self, 'bias'):
            output = output + T.forwardbroadcastitem(self.bias.bias(), X.ndim - 2)
        if hasattr(self, 'activation'):
            output = self.activation.apply(output)
        return output


class Pool(LayerBase):
    def __init__(self, windowsize=(2, 2), scale=False, bias=False, activation=None, stride=None, pad=None, mode='max',
                 **kwargs):
        LayerBase.__init__(self, **kwargs)
        self.windowsize = windowsize
        self.stride = stride
        self.pad = pad
        self.mode = mode
        self.scaled=scale
        self.biased=bias
        self.activated=activation

    def set_children(self):
        if self.scaled:
            self.scale = Scale()
        if self.biased:
            self.bias = Bias()
        if self.activated is not None:
            self.activation = Activation(self.activated)
        self.nchannels = self.in_dim[0]
        if hasattr(self, 'scale'):
            self.scale.unit_dim = self.nchannels
        if hasattr(self, 'bias'):
            self.bias.unit_dim = self.nchannels

    def apply(self, X):
        output = T.pool(X, self.windowsize(), mode=self.mode, stride=self.stride, pad=self.pad)
        if hasattr(self, 'scale'):
            output = output * T.forwardbroadcastitem(self.scale.scaleweight(), X.ndim - 2)
        if hasattr(self, 'bias'):
            output = output + T.forwardbroadcastitem(self.bias.bias(), X.ndim - 2)
        if hasattr(self, 'activation'):
            output = self.activation.apply(output)
        return output


class Flatten(LayerBase):
    def __init__(self, **kwargs):
        LayerBase.__init__(self, **kwargs)

    def apply(self, X):
        return T.tailflatten(X, 2)


class Asymconv(LayerBase):
    def __init__(self, asymfilters, filtersize, bias=False, activation=None, stride=None, pad=None, dilation=None,
                 **kwargs):
        LayerBase.__init__(**kwargs)
        self.asymfilters = asymfilters
        self.filtersize = filtersize
        self.stride = stride
        self.pad = pad
        self.dilation = dilation
        self.biased=bias
        self.activated=activation

    def set_children(self):
        if self.biased:
            self.bias = Bias()
        if self.activated is not None:
            self.activation = Activation(self.activated)
        self.nchannels = self.in_dim[0]
        self.imagesize = self.in_dim[1:]
        filters = []
        nfilters = 0

        def errorinfo(f):
            raise AssertionError(
                'filter not match channel, filter:%s   n_channel:%s' % (str(f), str(self.nchannels)))

        for filter in self.asymfilters:
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
        self.asymfilters = filters
        self.nfilters = nfilters
        filters = OrderedDict()
        for filter in self.asymfilters:
            for i in range(filter[2]):
                n = 'Filters' + '_' + str(filter[0]) + '_' + str(filter[1]) + '_' + str(i)
                filters[n] = [1, filter[0] * (self.nchannels / sum([filter[0], filter[1]]))] + self.filtersize
        self.filters = LayerDict(self, filters)

    def apply(self, X):
        outputs = []
        circle_shift = lambda choice, n: [i + 1 if i + 1 < n else 0 for i in choice]
        for filter in self.filters:
            choice = []
            for i in range(self.nchannels // sum([filter[0], filter[1]])):
                for j in range(filter[0]):
                    choice.append(i * sum([filter[0], filter[1]]) + j)
            for i in range(filter[2]):
                out = T.conv(X[:, choice], filter(), strides=self.stride,
                             pad=self.pad, dilation=self.dilation)
                outputs.append(out)
                choice = circle_shift(choice, self.nchannels)
        output = T.concatenate(outputs, axis=1)
        if hasattr(self, 'bias'):
            output = output + T.forwardbroadcastitem(self.bias.bias(), X.ndim - 2)
        if hasattr(self, 'activation'):
            output = self.activation.apply(output)
        return output
