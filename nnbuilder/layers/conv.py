# -*- coding: utf-8 -*-
"""
Created on  四月 26 21:08 2017

@author: aeloyq
"""

from nnbuilder import config
from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T

class ops:
    '''
    operations of layer's inner graph or model's cost function
    such as dropout\weight decay etc.
    '''

    class dropout:
        '''
        dropout
        '''
        name = 'dropout'
        use_noise = 'use_noise'

        def __init__(self):
            self.name = 'dropout'

        @staticmethod
        def op(tvar, **kwargs):
            return tvar * config.trng.binomial(tvar.shape,
                                               p=kwargs['use_noise'], n=1,
                                               dtype=tvar.dtype)

        @staticmethod
        def op_(tvar, **kwargs):
            return tvar * (1 - kwargs['use_noise'])

    class residual:
        '''
        residual
        '''
        name = 'residual'

        def __init__(self):
            self.name = 'residual'

        @staticmethod
        def op(tvar, **kwargs):
            return tvar + kwargs['pre_tvar']

        @staticmethod
        def op_(tvar, **kwargs):
            return ops.residual.op(tvar, **kwargs)

    class batch_normalization:
        '''
        batch normalization
        '''
        name = 'batch_normalization'

        def __init__(self):
            self.name = 'batch_normalization'

        @staticmethod
        def op(tvar, **kwargs):
            return T.nnet.batch_normalization(tvar, kwargs['gamma'], kwargs['beta'], kwargs['mean'], kwargs['std'])

        @staticmethod
        def op_(tvar, **kwargs):
            return ops.batch_normalization.op(tvar, **kwargs)