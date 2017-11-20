# -*- coding: utf-8 -*-
"""
Created on  四月 26 21:11 2017

@author: aeloyq
"""
from collections import OrderedDict
from nnbuilder.kernel import *

gated_activations = [T.glu, T.gtu, T.maxout]

class LayerDict(object):
    def __init__(self, layer, layers=None):
        '''

        :param layer:
        :param layers:
        '''
        self.layer = layer
        self.layers = OrderedDict()
        self.update(layers)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return self.layers[str(idx)]

    def __setitem__(self, idx, layer):
        return setattr(self, str(idx), layer)

    def __len__(self):
        return len(self.layers)

    def __iter__(self):
        return iter(self.layers.values())

    def __iadd__(self, layers):
        return self.update(layers)

    def __str__(self):
        keys = [str(layer) for layer in self.layers.values()]
        return keys

    def update(self, layers):
        '''

        :param layer:
        :return:
        '''
        self.layers.update(layers)
        for name, layer in layers:
            setattr(self.layer, name, layer)
        return self


class ParamDict(object):
    def __init__(self, layer, params=None):
        '''

        :param param:
        :param params:
        '''
        self.layer = layer
        self.params = OrderedDict()
        self.update(params)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError('index {} is out of range'.format(idx))
        if idx < 0:
            idx += len(self)
        return self.params[str(idx)]

    def __setitem__(self, idx, param):
        return setattr(self, str(idx), param)

    def __len__(self):
        return len(self.params)

    def __iter__(self):
        return iter(self.params.values())

    def __iadd__(self, params):
        return self.update(params)

    def __str__(self):
        keys = [str(param) for param in self.params.values()]
        return keys

    def update(self, params):
        '''

        :param param:
        :return:
        '''
        self.params.update(params)
        for name, param in params:
            setattr(self.layer, name, param)
        return self
