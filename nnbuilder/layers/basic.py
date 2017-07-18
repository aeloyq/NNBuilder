# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:55:42 2016

@author: aeloyq
"""

from nnbuilder import config
from collections import OrderedDict
from roles import weight, bias
from utils import *
from ops import *
import numpy as np
import theano
import theano.tensor as T
import copy


class baselayer:
    '''
     base class of layer
    '''

    def __init__(self, **kwargs):
        '''
        initiate the layer class to an instance
        :param kwargs: 
        '''
        self.kwargs = kwargs
        self.rng = config.rng
        self.trng = config.trng
        self.name = 'None'
        self.children_name = ''
        self.input = None
        self.output = None
        self.children = OrderedDict()
        self.params = OrderedDict()
        self.P = OrderedDict()
        self.inited_param = False
        self.inited_children = False
        self.input_set = False
        self.roles = OrderedDict()
        self.updates = OrderedDict()
        self.raw_updates = OrderedDict()
        self.ops = OrderedDict()
        self.debug_stream = []
        self.x_mask = None
        self.y_mask = None
        self.y = None
        self.data = OrderedDict()
        self.setattr('name')
        self.setattr('input')
        self.setattr('output')
        self.setattr('children')
        self.setattr('params')
        self.setattr('roles')
        self.setattr('updates')
        self.setattr('ops')
        self.setattr('debug_stream')
        self.setattr('data')
        self.in_dim_set = False
        self.setattr('in_dim')
        if 'in_dim' in kwargs: self.in_dim_set = True

    def set_children(self):
        '''
        set the children of this layer
        :return: None
        '''
        pass

    def set_in_dim(self, dim):
        if not self.in_dim_set:
            self.in_dim = dim
            self.in_dim_set = True

    def set_name(self, name):
        '''
        set the name of layer
        :param name: str
            name of layer
        :return: None
        '''
        self.name = name

    def set_input(self, X):
        '''
        set the input of layer
        :param X: tensor variable
            input tensor
        :return: None
        '''
        self.input = X

    def set_mask(self, x_mask, y_mask):
        '''
        set the input of layer
        :param X: tensor variable
            input tensor
        :return: None
        '''
        self.x_mask = x_mask
        self.y_mask = y_mask

    def output_ops(self):
        '''
        public ops on tensor variable
        such like dropout batch normalization etc.
        if want to change the sequence of ops
        please overwrite this function
        :return: None
        '''
        pass

    def feedforward_ops(self):
        '''
        public ops on tensor variable
        such like dropout batch normalization etc.
        if want to change the sequence of ops
        please overwrite this function
        :return: None
        '''
        pass

    def build_ops(self):
        '''
        public ops on tensor variable
        such like dropout batch normalization etc.
        if want to change the sequence of ops
        please overwrite this function
        :return: None
        '''
        self.output = self.addops('output', self.output, layernorm)
        self.output = self.addops('output', self.output, dropout)

    def addops(self, name, tvar, ops, switch=True, **options):
        '''
        add operation on tensor variable
        which realize training tricks
        such as dropout residual etc.
        :param name: str
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
        name = self.name + '_' + name + '_' + ops.name
        if name not in self.ops: self.ops[name] = switch
        if not self.ops[name]: return tvar
        if name in self.op_dict:
            dict = self.op_dict[name]
        else:
            if ops.name not in self.op_dict: return tvar
            dict = self.op_dict[ops.name]
        dict=copy.deepcopy(dict)
        dict.update(options)
        if self.op_dict['mode'] == 'train':
            return ops.op(tvar, **dict)
        elif self.op_dict['mode'] == 'use':
            return ops.op_(tvar, **dict)

    def initiate(self, name, dim=None, **op_dict):
        '''
        initiate the layer
        :param dim: int
            dim of the input
        :param name: str
            name of the layer    
        :return: None
        '''
        self.set_in_dim(dim)
        self.op_dict = op_dict
        self.set_name(name)
        self.updates = OrderedDict()
        self.raw_updates = OrderedDict()
        if not self.inited_param:
            self.init_params()
            self.inited_param = True
        if not self.inited_children:
            self.set_children()
            self.inited_children = True

        self.init_children()

    def init_params(self):
        '''
        initiate the params
        :return: None
        '''
        pass

    def allocate(self, rndfn, name, role, *args):
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
        if name in self.kwargs:
            if callable(self.kwargs[name]):
                rndfn = self.kwargs[name]
            else:
                pname = name + '_' + self.name
                self.params[pname] = theano.shared(value=self.kwargs[name], name=pname, borrow=True)
                self.P[name] = self.params[pname]
                self.roles[pname] = role
                return self.params[pname]
        pname = name + '_' + self.name
        self.params[pname] = theano.shared(value=rndfn(*args), name=pname, borrow=True)
        self.P[name] = self.addops(pname, self.params[pname], weightnorm)
        self.roles[pname] = role
        return self.P[name]

    def init_children(self):
        '''
        initiate the children to the layer
        :return: None
        '''
        for name, child in self.children.items():
            child.set_in_dim(self.in_dim)
            child.children_name = name
            child.data = self.data
            child.x_mask = self.x_mask
            child.y_mask = self.y_mask
            child.y = self.y
            child.initiate(self.name + '_' + name, **self.op_dict)
        self.merge(0)

    def merge(self, step=0):
        '''
        merge the children to the layer
        :return: None
        '''
        for name, child in self.children.items():
            if step == 0:
                self.params.update(child.params)
                self.roles.update(child.roles)
            if step == 1:
                self.ops.update(child.ops)
                self.updates.update(child.updates)

    def feedforward(self, X=None, P=None):
        '''
        feed forward to get the graph
        :return: tensor variable
            get the graph
        '''
        if X is None:
            X = self.input
        if P == None or P == {}:
            P = self.P
        else:
            if self.children_name in P:
                dict = self.P
                dict.update(P[self.children_name])
                P = dict
        self.set_input(X)
        self.get_output(X, P)
        self.feedforward_ops()
        self.merge(1)
        return self.output

    def get_output(self, X, P):
        '''
        get the output of layer
        :return: tensor variable
            output of layer
        '''
        self.output = self.apply(X, P)
        self.output_ops()

    def apply(self, X, P):
        '''
        build the graph of layer
        :return: tensor variable
            the computational graph of this layer
        '''
        return X

    def build(self, input, name, **op_dict):
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
        self.initiate(name, **op_dict)
        self.feedforward(input)
        self.build_ops()
        return self.output

    def evaluate(self, input, name, **op_dict):
        '''
        evaluate for model
        :param input: class on the base of baselayer or tensorvariable
            the input of the layer
        :param dim: int
            dim of the input  
        :param name: str
            the name of the layer
        :param op_dict: OrderedDict
            operation dictionary of the layer
            contains extra operations
            include:
                dropout
                mask
                etc.
        :return: 
        '''
        if isinstance(input, baselayer):
            input = input.output
        else:
            input = input
        self.build(input, name, **op_dict)

    def setattr(self, name):
        '''
        set the attribute of the layer class
        :param name: str
            name of attribute
        :return: None
        '''
        if name in self.kwargs:
            setattr(self, name, self.kwargs[name])

    def debug(self, *variable):
        '''
        add tensor variable to debug stream
        :param variable: tensor variable
            tensor variable which want to be debug in debug mode
        :return: None
        '''
        self.debug_stream.extend(list(variable))


class layer(baselayer):
    '''
    abstract layer
    '''

    def __init__(self, unit, **kwargs):
        baselayer.__init__(self, **kwargs)
        self.unit_dim = unit

    def apply(self, X, P):
        if len(self.children) > 1:
            feedlist = []
            for name, child in self.children.items():
                chd = child
                input = X
                feedlist.append(chd.feedforward(input, P))
            return concatenate(feedlist, axis=feedlist[0].ndim - 1)
        elif len(self.children) == 1:
            child = self.children.items()[0][1]
            chd = child
            input = X
            return chd.feedforward(input, P)
        else:
            return X


class linear(layer):
    '''
    linear layer
    '''

    def __init__(self, unit, activation=None, **kwargs):
        layer.__init__(self, unit, **kwargs)
        self.activation = activation

    def init_params(self):
        self.wt = self.allocate(uniform, 'Wt', weight, self.in_dim, self.unit_dim)

    def apply(self, X, P):
        return T.dot(X, P['Wt'])

    def output_ops(self):
        if self.activation is not None:
            self.output = self.activation(self.output)



class std(linear):
    '''
    std layer
    '''

    def init_params(self):
        self.wt = self.allocate(randn, 'Wt', weight, self.in_dim, self.unit_dim)


class linear_bias(linear):
    '''
    linear layer with bias
    '''

    def __init__(self, unit, activation=None, **kwargs):
        linear.__init__(self, unit, activation, **kwargs)

    def init_params(self):
        linear.init_params(self)
        self.bi = self.allocate(zeros, 'Bi', bias, self.unit_dim)

    def apply(self, X, P):
        return T.dot(X, P['Wt']) + P['Bi']


class hidden_layer(layer):
    '''
    setup base hidden layer
    '''

    def __init__(self, unit, activation=T.tanh, **kwargs):
        layer.__init__(self, unit, **kwargs)
        self.activation = activation

    def set_children(self):
        self.children['lb'] = linear_bias(self.unit_dim, self.activation)


class output_layer(layer):
    ''' 
    setup base output layer 
    '''

    def __init__(self, unit, activation=T.nnet.sigmoid, **kwargs):
        layer.__init__(self, unit, **kwargs)
        self.activation = activation
        self.cost_function = mean_square
        self.cost = None
        self.predict = None
        self.error = None
        self.setattr('cost_function')
        self.setattr('cost')
        self.setattr('predict')
        self.setattr('error')

    def set_children(self):
        self.children['lb'] = linear_bias(self.unit_dim, self.activation)

    def get_predict(self):
        '''
        get the predict of the model
        :return: tensor variable
            the predict of model
        '''
        self.predict = T.round(self.output)

    def get_cost(self, Y):
        '''
        get the cost of the model
        :param Y: 
            the label of the model which used to evaluate the cost function(loss function)
        :return: tensor variable
            the cost of the model
        '''
        self.cost = self.cost_function(Y, self.output)

    def get_error(self, Y):
        '''
        get the error of the model
        :param Y: 
            the label of the model which used to caculate the error
        :return: tensor variable
            the error (1-accruate%) of the model
        '''
        self.error = T.mean(T.neq(Y, self.predict))
