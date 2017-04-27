# -*- coding: utf-8 -*-
"""
Created on  四月 26 21:10 2017

@author: aeloyq
"""

from nnbuilder import config
from roles import *
from collections import OrderedDict
from basic import baselayer
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

    def init(self,layer,ops_dict):
        self.layer=layer
        ops_dict[layer].append(self)

    @staticmethod
    def op(tvar, **kwargs):
        pass

    @staticmethod
    def op_(tvar, **kwargs):
        pass

    def evaluate(self,**kwargs):
        pass

class dropout(ops):
    name='dropout'
    def __init__(self,  noise=0.5,gate=None):
        ops.__init__(self)
        # if dp_name != None and use_noise#Todo:multi setting of noise
        self.noise=noise
        self.dp_name=gate
        self.noise = theano.shared(value=self.noise, name='dropout_noise', borrow=True)


    @staticmethod
    def op(tvar, **kwargs):
        return tvar * config.trng.binomial(tvar.shape,
                                           p=kwargs['use_noise'], n=1,
                                           dtype=tvar.dtype)

    @staticmethod
    def op_(tvar, **kwargs):
        return tvar * (1 - kwargs['use_noise'])

    def evaluate(self):
        if self.dp_name != None:
            self.dp_name = [name[0] + '_' + name for name in self.dp_name]
        else:
            self.dp_name = None
        if self.dp_name == None:
            self.op_dict = {'dropout': {'use_noise': self.noise}}
        else:
            self.op_dict = {'dropout': {'use_noise': self.noise}}
            for name in self.layer.ops:
                self.layer.ops[name] = False
            for name in self.dp_name:
                self.layer.ops[name] = True
                self.op_dict[name] = {'use_noise': self.noise}


class weight_decay(ops):
    def __init__(self,noise=0.0001,params=None):
        ops.__init__(self)
        self.l2 = noise
        self.params = params

    def init(self,layer,ops_dict):
        self.layer=layer
        ops_dict['cost'].append(self)

    def evaluate(self, cost):
        reg = 0
        params=[]
        pdict=OrderedDict()
        if self.params==None:
            pdict=self.layer.params
        else:
            for pname in params:
                for name,param in self.layer.params.items():
                    if pname==name.split('_')[0]:
                        pdict[name]=param

        for name, param in pdict.items():
            if self.layer.roles[name] == weight:
                params.append(param)
        for param in params:
            reg += (param ** 2).sum()
        return cost + self.l2 * reg

class regularization(ops):
    def __init__(self,noise=0.0001):
        ops.__init__(self)
        self.l2 = noise

    def init(self,layer,ops_dict):
        self.layer=layer
        ops_dict['cost'].append(self)

    def evaluate(self, cost ,graph):
        reg = 0
        params=[]
        for node in graph:
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
        self.op_dict['pre_tvar'] = self.pre_layer.output

    @staticmethod
    def op(tvar, **kwargs):
        return tvar + kwargs['pre_tvar']

    @staticmethod
    def op_(tvar, **kwargs):
        return residual.op(tvar, **kwargs)

class batch_normalization(ops):
    def __init__(self):
        ops.__init__(self)
        self.op_dict = {}

    def evaluate(self):
        self.op_dict['pre_tvar'] = None

    @staticmethod
    def op(tvar, **kwargs):
        return T.nnet.batch_normalization(tvar, kwargs['gamma'], kwargs['beta'], kwargs['mean'], kwargs['std'])

    @staticmethod
    def op_(tvar, **kwargs):
        return batch_normalization.op(tvar, **kwargs)