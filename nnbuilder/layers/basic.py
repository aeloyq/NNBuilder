# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:55:42 2016

@author: aeloyq
"""

import copy
import numpy as np
from utils import *
from nnbuilder.kernel import *
from collections import OrderedDict
from roles import *
from ops import *


class base(object):
    '''
     base class of layer
    '''

    def __init__(self, **kwargs):
        '''
        initiate the layer class to an instance
        :param kwargs: 
        '''
        super(base, self).__init__()
        self.kwargs = kwargs
        self.name = None
        self.pre_layer = None
        self.next_layer = None
        self.in_dim = None
        self.out_dim = None
        self.input = None
        self.output = None
        self.raw_output = None
        self.data = OrderedDict()
        self.params = OrderedDict()
        self.roles = OrderedDict()
        self.shapes = OrderedDict()
        self.rndfns = OrderedDict()
        self.trainable_params = OrderedDict()
        self.trainable_roles = OrderedDict()
        self.trainable_shapes = OrderedDict()
        self.trainable_rndfns = OrderedDict()
        self.untrainable_params = OrderedDict()
        self.updates = OrderedDict()
        self.raw_updates = OrderedDict()
        self.ops = []
        self.ops_option = OrderedDict()
        self.shared_info = OrderedDict()
        self.debug_stream = []
        self.setattr('name')
        self.setattr('in_dim')
        self.setattr('input')

    def setattr(self, name, attr=None):
        '''
        set the attribute of the layer class
        :param name: str
            name of attribute
        :return: None
        '''
        if name in self.kwargs:
            setattr(self, name, self.kwargs[name])
        else:
            if attr is not None:
                setattr(self, name, attr)

    def set_name(self, name):
        '''
        set the name of layer
        :param name: str
            name of layer
        :return: None
        '''
        self.name = name

    def set_pre_layer(self, pre_layer):
        self.pre_layer = pre_layer

    def set_raw_output(self, raw_output):
        self.raw_output = raw_output

    def set_in_dim(self, dim):
        if isinstance(dim, tuple):
            dim = list(dim)
        self.in_dim = dim

    def get_out_dim(self):
        self.out_dim = self.in_dim

    def set_input(self, X):
        '''
        set the input of layer
        :param X: tensor variable
            input tensor
        :return: None
        '''
        self.input = X

    def set_data(self, data):
        self.data = data

    def init_params(self):
        '''
        initiate the params
        :return: None
        '''
        pass

    def allocate(self, rndfn, name, role, shape, **kwargs):
        '''
        allocate the param to the ram of gpu(cpu)
        :param role: roles
            sub class of roles
        :param name: str
            name of param
        :param rndfn:
            initiate function of param
        :param args:
            the shape of param
        :return: theano shared variable
            the allocated param
        '''
        pname = self.name + '_' + name
        if pname not in self.trainable_params:
            if name in self.kwargs:
                if callable(self.kwargs[name]):
                    rndfn = self.kwargs[name]
                else:
                    self.trainable_params[pname] = kernel.shared(value=self.kwargs[name], name=pname, attr=role.attr)
                    self.params[name] = self.trainable_params[pname]
                    self.roles[name] = role
                    self.rndfns[name] = self.kwargs[name]
                    self.shapes[name] = shape
                    self.trainable_roles[pname] = role
                    self.trainable_rndfns[pname] = self.kwargs[name]
                    self.trainable_shapes[pname] = shape
                    return self.trainable_params[pname]
            self.trainable_params[pname] = kernel.shared(value=rndfn(shape, **kwargs), name=pname, attr=role.attr)
            self.params[name] = self.trainable_params[pname]
            self.roles[name] = role
            self.rndfns[name] = rndfn
            self.shapes[name] = shape
            self.trainable_roles[pname] = role
            self.trainable_rndfns[pname] = rndfn
            self.trainable_shapes[pname] = shape
        return self.trainable_params[pname]

    def apply_ops(self, name, tvar, ops, **extra_options):
        '''
        add operation on tensor variable
        which realize training tricks
        such as dropout residual etc.
        :param opname: str
            name of operation
        :param tvar: tensor variable
            tensor variable on which the operation add
        :param ops: class ops
            kind of operation
        :param switch: bool
            open or close the operation for default
        :return: callable
            the operation function
        '''
        opname = name + '_' + ops.name
        if ops.name not in self.ops:
            if opname not in self.ops: return tvar
        if name in self.ops_option[ops.name]:
            option = self.ops_option[ops.name][name]
        else:
            option = self.ops_option[ops.name]['default']
        option = copy.deepcopy(option)
        option.update(extra_options)
        option['oname'] = name
        if self.ops_option['mode'] == 'train':
            return ops.op(self, tvar, **option)
        elif self.ops_option['mode'] == 'use':
            return ops.op_(self, tvar, **option)

    def apply(self, X):
        '''
        build the graph of layer
        :return: tensor variable
            the computational graph of this layer
        '''
        return X

    def initiate(self, ops_option):
        '''
        initiate the layer
        :param dim: int
            dim of the input
        :param name: str
            name of the layer
        :return: None
        '''
        self.ops_option = ops_option

    def feed(self, X):
        '''
        feed forward to get the graph
        :return: tensor variable
            get the graph
        '''
        output = self.apply(X)
        return output

    def build(self, ops_option):
        '''
        build the layer
        :param input: tensor variable
            input of layer
        :param dim: int
            dim of the input
        :param name:
            name of layer
        :return: tensor variable
            output of layer
        '''
        self.set_input(self.pre_layer.output)
        self.initiate(ops_option)
        self.output = self.feed(self.input)
        return self.output

    def pre_build(self, name, prelayer, data):
        self.set_name(name)
        self.set_pre_layer(prelayer)
        self.set_data(data)
        self.set_in_dim(prelayer.out_dim)
        self.get_out_dim()
        self.init_params()


class entity(base):
    def debug(self, *variable):
        '''
        add tensor variable to debug stream
        :param variable: tensor variable
            tensor variable which want to be debug in debug mode
        :return: None
        '''
        self.debug_stream.extend(list(variable))


class component(base):
    def __init__(self, **kwargs):
        base.__init__(self, **kwargs)


class forward(base):
    '''
    abstract layer
    '''

    def __init__(self, unit, **kwargs):
        base.__init__(self, **kwargs)
        self.unit_dim = unit

    def get_out_dim(self):
        self.out_dim = self.unit_dim


class integrate(base):
    '''
    abstract layer
    '''

    def __init__(self, **kwargs):
        base.__init__(self, **kwargs)


class transfer(base):
    '''
    abstract layer
    '''

    def __init__(self, **kwargs):
        base.__init__(self, **kwargs)


class hidden(base):
    '''
    setup base hidden layer
    '''

    def __init__(self, **kwargs):
        base.__init__(self, **kwargs)


class output(base):
    '''
    setup base output layer
    '''

    def __init__(self, **kwargs):
        base.__init__(self, **kwargs)
        self.loss = None
        self.sample = None
        self.sample_loss = None
        self.sample_error = None
        self.predict = None
        self.loss_function = loss_functions.mse
        self.setattr('loss_function')

    def apply_loss(self, Y_True):
        '''
        get the cost of the model
        :param Y_True:
            the label of the model which used to evaluate the cost function(loss function)
        :return: tensor variable
            the cost of the model
        '''
        return self.loss_function(self.output, Y_True)

    def apply_sample(self):
        '''
        get the predict of the model
        :return: tensor variable
            the predict of model
        '''
        return T.round(self.output)

    def apply_sample_loss(self, Y_True):
        return self.apply_loss(Y_True)

    def apply_sample_error(self, Y_True):
        return T.mean(T.neq(self.sample, Y_True))

    def apply_predict(self):
        return self.apply_sample()


class linear(component):
    '''
    linear layer
    '''

    def __init__(self, **kwargs):
        component.__init__(self, **kwargs)

    def init_linear_params(self, in_dim, unit_dim, biased, weight_name='Wt', weight_role=weight,
                           init_functions=(param_init_functions.uniform, param_init_functions.zeros), unit_name=''):
        self.allocate(init_functions[0], unit_name + weight_name, weight_role, [in_dim, unit_dim])
        if biased:
            self.allocate(init_functions[1], unit_name + 'Bi', bias, [unit_dim])

    def layer_dot(self, name, X, W='Wt'):
        unit_dim = self.shapes[name + W][-1]
        X = self.apply_ops(name=name, tvar=X, ops=dropout)
        w = self.apply_ops(name=name, tvar=self.params[name + W], ops=weightnorm)
        o = T.dot(X, w)
        o = self.apply_ops(name=name, tvar=o, ops=normalization, unit_dim=unit_dim)
        return o

    def apply_bias(self, name, tvar, biased):
        if biased:
            return tvar + self.params[name + 'Bi']
        else:
            return tvar

    def apply_activation(self, tvar, activation):
        if activation is not None:
            return activation(tvar)
        else:
            return tvar


class sparse(base):
    pass


class fwdlinear(forward, linear):
    units_name = 'Feedforward'

    def __init__(self, unit, biased=True, activation=None, **kwargs):
        forward.__init__(self, unit, **kwargs)
        self.biased = biased
        self.activation = activation
        self.units = [fwdlinear.units_name]
        self.units_dim = {fwdlinear.units_name: unit}

    def init_params(self):
        self.init_linear_params(self.in_dim, self.unit_dim, self.biased, unit_name=self.units[0])

    def apply(self, X):
        o = self.layer_dot(name=self.units[0], X=X)
        o = self.apply_bias(name=self.units[0], tvar=o, biased=self.biased)
        o = self.apply_activation(o, self.activation)
        return o


class itglinear(integrate, linear):
    def __init__(self, **kwargs):
        integrate.__init__(self, **kwargs)

    def init_all_linear_params(self, units_dim, biased):
        for name, unit in units_dim.items():
            if not isinstance(unit, (tuple, list)):
                self.init_linear_params(self.in_dim, unit, biased, unit_name=name)
            elif not isinstance(unit[0], (tuple, list)):
                self.init_linear_params(unit[0], unit[1], biased, unit_name=name)
            else:
                self.init_linear_params(unit[0][0], unit[0][1], unit[1], unit_name=name)


class lookuptable(component):
    def __init__(self, unit, **kwargs):
        component.__init__(self, **kwargs)
        self.emb_dim = unit

    def init_lookuptable_params(self, vocab_dim, emb_dim, weight_name='Lookuptable', unit_name=''):
        self.allocate(param_init_functions.randn, unit_name + weight_name, weight, [vocab_dim, emb_dim])

    def layer_lookup(self, name, X, D='Lookuptable'):
        embedding = T.lookup(X, self.params[name + D])
        embedding = self.apply_ops(name, tvar=embedding, ops=dropout, shape=X.shape, broadcast=-1)
        return embedding


class fwdlookup(forward, lookuptable):
    units_name = ['SourceWord']

    def __init__(self, unit, **kwargs):
        forward.__init__(self, unit, **kwargs)
        lookuptable.__init__(self, unit, **kwargs)
        self.units = [fwdlookup.units_name[0]]
        self.units_dim = {fwdlookup.units_name[0]: unit}

    def set_in_dim(self, dim):
        forward.set_in_dim(self, dim)
        self.vocab_dim = self.in_dim

    def init_params(self):
        self.init_lookuptable_params(self.vocab_dim, self.emb_dim, unit_name=self.units[0])

    def apply(self, X):
        return self.layer_lookup(name=self.units[0], X=X)

    def get_out_dim(self):
        self.out_dim = self.emb_dim


class itglookup(integrate, lookuptable):
    def __init__(self, **kwargs):
        integrate.__init__(self, **kwargs)

    def init_all_lookup_params(self, units_dim):
        for name, unit in units_dim.items():
            if not isinstance(unit, (tuple, list)):
                self.init_lookuptable_params(self.in_dim, unit, unit_name=name)
            else:
                self.init_lookuptable_params(unit[0], unit[1], unit_name=name)
