# -*- coding: utf-8 -*-
"""
Created on  四月 26 21:10 2017

@author: aeloyq
"""

from nnbuilder import config
from roles import *
from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T


class ops:
    '''
    operations of layer's inner graph or model's cost function
    such as dropout\weight decay etc.
    '''

    def __init__(self):
        pass

    def init(self, layer, ops_dict):
        self.layer = layer
        ops_dict[layer].append(self)

    @staticmethod
    def op(tvar, **kwargs):
        pass

    @staticmethod
    def op_(tvar, **kwargs):
        pass

    def evaluate(self, **kwargs):
        pass


class dropout(ops):
    name = 'dropout'

    def __init__(self, noise=0.5, gate=None):
        ops.__init__(self)
        self.noise = noise
        self.gate = gate

    @staticmethod
    def op(tvar, **kwargs):
        if 'broadcast' not in kwargs:kwargs['broadcast']=-1
        if 'rev_sec' not in kwargs:kwargs['rev_sec']=False
        if not isinstance(tvar,list):
            shape = tvar.shape
            if kwargs['broadcast'] != -1:
                shape = []
                for i in range(-kwargs['broadcast'] - 1, -1, -1):
                    shape.append(tvar.shape[i])
            return tvar * config.trng.binomial(shape,
                                               p=kwargs['use_noise'], n=1,
                                               dtype='float32')/kwargs['use_noise']
        else:
            tvarlist=[]
            for idx,tvr in enumerate(tvar):
                shape=tvr.shape
                if kwargs['broadcast']!=-1:
                    shape=[]
                    for i in range(-kwargs['broadcast']-1,-1,-1):
                        shape.append(tvr.shape[i])
                if idx%2!=0 or not kwargs['rev_sec']:
                    tvarlist.append(tvr * config.trng.binomial(shape,
                                                   p=kwargs['use_noise'], n=1,
                                                   dtype='float32')/kwargs['use_noise'])
                else:
                    tvarlist.append(tvr * config.trng.binomial(shape,
                                                               p=kwargs['use_noise'], n=1,
                                                               dtype='float32')[::-1]/kwargs['use_noise'])
            return tvarlist

    @staticmethod
    def op_(tvar, **kwargs):
        if not isinstance(tvar,list):
            return tvar
        else:
            tvarlist = []
            for idx, tvr in enumerate(tvar):
                tvarlist.append(tvr)
            return tvarlist

    def evaluate(self):
        if self.gate != None:
            self.gate = [name[0] + '_' + name for name in self.gate]
            self.op_dict = {'dropout': {'use_noise': self.noise}}
            for name in self.layer.ops:
                self.layer.ops[name] = False
            for name in self.gate:
                self.layer.ops[name] = True
                self.op_dict[name] = {'use_noise': self.noise}
        else:
            self.op_dict = {'dropout': {'use_noise': self.noise}}


class weight_decay(ops):
    def __init__(self, noise=0.0001, params=None):
        ops.__init__(self)
        self.l2 = noise
        self.params = params

    def init(self, layer, ops_dict):
        self.layer = layer
        ops_dict['cost'].append(self)

    def evaluate(self, cost):
        reg = 0
        params = []
        pdict = OrderedDict()
        if self.params == None:
            pdict = self.layer.params
        else:
            for pname in params:
                for name, param in self.layer.params.items():
                    if pname == name.split('_')[0]:
                        pdict[name] = param

        for name, param in pdict.items():
            if self.layer.roles[name] == weight:
                params.append(param)
        for param in params:
            reg += (param ** 2).sum()
        return cost + self.l2 * reg


class regularization(ops):
    def __init__(self, noise=0.0001):
        ops.__init__(self)
        self.l2 = noise

    def init(self, layer, ops_dict):
        self.layer = layer
        ops_dict['cost'].append(self)

    def evaluate(self, cost, layers):
        reg = 0.
        params = []
        for lname, node in layers.items():
            for name, param in node.params.items():
                if node.roles[name] == weight:
                    params.append(param)
        for param in params:
            reg += (param ** 2).sum()
        return cost + self.l2 * reg


class residual(ops):
    def __init__(self, pre_layer):
        ops.__init__(self)
        self.pre_layer = pre_layer
        self.op_dict = {}

    def evaluate(self):
        self.op_dict['residual'] = {'pre_tvar':self.pre_layer.output}

    @staticmethod
    def op(tvar, **kwargs):
        return tvar + kwargs['pre_tvar']

    @staticmethod
    def op_(tvar, **kwargs):
        return residual.op(tvar, **kwargs)


class layernorm(ops):
    name = 'layernorm'

    def __init__(self, beta=0.0, gamma=1.0):
        ops.__init__(self)
        self.op_dict = {}
        self.beta = beta
        self.gamma = gamma

    def evaluate(self):
        self.op_dict['layernorm'] ={'beta':self.beta,'gamma':self.gamma}

    @staticmethod
    def op(tvar, **kwargs):
        _eps = 1e-5
        output = (tvar - tvar.mean(-1, keepdims=True)) / T.sqrt((tvar.var(-1, keepdims=True) + _eps))
        output = kwargs['gamma'] * output + kwargs['beta']
        return output

    @staticmethod
    def op_(tvar, **kwargs):
        return layernorm.op(tvar, **kwargs)


class weightnorm(ops):
    name = 'weightnorm'

    def __init__(self, gamma=1.0):
        ops.__init__(self)
        self.op_dict = {}
        self.gamma = gamma

    def evaluate(self):
        self.op_dict['weightnorm'] ={'gamma':self.gamma}

    @staticmethod
    def op(tvar, **kwargs):
        _eps = 1e-5
        norm = T.sqrt((tvar * tvar).sum(axis=0, keepdims=True) + _eps)
        norm_weight = tvar / (norm * kwargs['gamma'])
        return norm_weight

    @staticmethod
    def op_(tvar, **kwargs):
        return weightnorm.op(tvar, **kwargs)
