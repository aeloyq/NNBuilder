# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:55:42 2016

@author: aeloyq
"""

import numpy as np
from utils import *
from basic import *
from roles import *
from ops import *
from nnbuilder.kernel import *


class fwdhidden(fwdlinear, hidden):
    ''' 
    setup hidden layer of feedforward network inherited from Hidden_Layer
    '''
    pass


class fwdoutput(fwdlinear, output):
    pass


class gtdhidden(fwdhidden):
    '''
    gated hidden layer
    '''

    def __init__(self, unit, biased=True, gate=T.glu, **kwargs):
        super(gtdhidden, self).__init__(unit=unit, biased=biased, activation=gate, **kwargs)
        self.unit_dim = unit * 2
        self.gate = gate
        if gate not in utils.gated_activation:
            raise AssertionError('Given gate is not a gated activation. gate:%s  gated activation:%s' % (
                str(gate), str(utils.gated_activation)))


class trfhidden(transfer, hidden):
    pass


class trfoutput(transfer, output):
    pass


class lkphidden(fwdlookup, hidden):
    pass


class multilkphidden(itglookup, hidden):
    def __init__(self, units, **kwargs):
        itglookup.__init__(**kwargs)
        self.units = units
        self.units_dim = units.copy()

    def init_params(self):
        self.init_all_lookup_params(self.units_dim)

    def get_out_dim(self):
        self.out_dim = sum(self.units_dim.values())

    def apply(self, X):
        return T.concatenate([self.layer_lookup(name=name, X=X[:, :, i]) for i, name in enumerate(self.units)],
                             axis=X.ndim)


class hnn(fwdhidden, entity):
    pass


class gnn(gtdhidden, entity):
    pass


class embedding(lkphidden, entity):
    pass


class multiembedding(multilkphidden, entity):
    pass


class activation(trfhidden, entity):
    def __init__(self, activation=T.tanh, **kwargs):
        trfhidden.__init__(**kwargs)
        self.activation = activation

    def apply(self, X):
        return self.activation(X)

    def get_out_dim(self):
        if self.activation not in utils.gated_activation:
            self.out_dim = self.in_dim
        else:
            self.out_dim = self.in_dim / 2


class direct(trfoutput, entity):
    ''' setup direct output layer inherited from base output layer '''
    pass


class logistic(fwdoutput, entity):
    def __init__(self, unit, biased=True, **kwargs):
        fwdoutput.__init__(self, unit=unit, biased=biased, activation=T.sigmoid, **kwargs)
        output.__init__(self, loss_function=loss_functions.ce, **kwargs)


class softmax(fwdoutput, entity):
    def __init__(self, unit, biased=True, **kwargs):
        fwdlinear.__init__(self, unit=unit, biased=biased, activation=T.softmax,
                           **kwargs)
        output.__init__(self, loss_function=loss_functions.nlog, **kwargs)

    def apply_sample(self):
        return self.output.argmax(-1)
