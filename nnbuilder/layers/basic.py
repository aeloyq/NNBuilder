# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:55:42 2016

@author: aeloyq
"""

from nnbuilder import config
from collections import OrderedDict
from roles import weight,bias
from utils import *
from ops import *
import numpy as np
import theano
import theano.tensor as T

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
        self.name = 'None'
        self.input = None
        self.output = None
        self.children = OrderedDict()
        self.params = OrderedDict()
        self.inited_param = False
        self.roles = OrderedDict()
        self.updates = OrderedDict()
        self.ops = OrderedDict()
        self.debug_stream = []
        self.setattr('name')
        self.setattr('input')
        self.setattr('output')
        self.setattr('children')
        self.setattr('params')
        self.setattr('roles')
        self.setattr('updates')
        self.setattr('ops')
        self.setattr('debug_stream')

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

    def get_output(self):
        '''
        get the output of layer
        :return: tensor variable
            output of layer
        '''
        self.output = self.apply()
        self.public_ops()

    def public_ops(self):
        '''
        public ops on tensor variable
        such like dropout batch normalization etc.
        if want to change the sequence of ops
        please overwrite this function
        :return: None
        '''
        self.output = self.addops('output', self.output, dropout)

    def addops(self, name, tvar, ops, switch=True):
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
        name = name + '_' + ops.name
        if name not in self.ops: self.ops[name] = switch
        if not self.ops[name]: return tvar
        if name in self.op_dict:
            dict = self.op_dict[name]
        else:
            if ops.name not in self.op_dict: return tvar
            dict = self.op_dict[ops.name]
        if self.op_dict['mode'] == 'train':
            return ops.op(tvar, **dict)
        elif self.op_dict['mode'] == 'use':
            return ops.op_(tvar, **dict)

    def apply(self):
        '''
        build the graph of layer
        :return: tensor variable
            the computational graph of this layer
        '''
        return self.input

    def _allocate(self, rndfn, name, role, *args):
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
                name = name + '_' + self.name
                self.params[name] = theano.shared(value=self.kwargs[name], name=name, borrow=True)
                self.roles[name] = role
                return self.params[name]
        name = name + '_' + self.name
        self.params[name] = theano.shared(value=rndfn(*args), name=name, borrow=True)
        self.roles[name] = role
        return self.params[name]

    def init_params(self):
        '''
        initiate the params
        :return: None
        '''
        pass

    def initiate(self,dim, name, **op_dict):
        '''
        initiate the layer
        :param dim: int
            dim of the input
        :param name: str
            name of the layer    
        :return: None
        '''
        self.in_dim=dim
        self.op_dict = op_dict
        self.set_name(name)
        if not self.inited_param:
            self.init_params()
            self.inited_param = True

    def feedforward(self, X):
        '''
        feed forward to get the graph
        :return: tensor variable
            get the graph
        '''
        self.set_input(X)
        self.get_output()
        return self.output

    def build(self, input, dim, name, **op_dict):
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
        self.initiate(dim,name, **op_dict)
        self.init_children()
        self.feedforward(input)
        self.merge()
        return self.output

    def init_children(self):
        '''
        initiate the children to the layer
        :return: None
        '''
        for name, child in self.children.items():
            chd = child
            if isinstance(child, list):
                chd = child[0]
            chd.ops['output'] = False
            chd.initiate(self.in_dim,self.name + '_' + name, **self.op_dict)

    def merge(self):
        '''
        merge the children to the layer
        :return: None
        '''
        for name, child in self.children.items():
            chd = child
            if isinstance(child, list):
                chd = child[0]
            self.params.update(chd.params)
            self.roles.update(chd.roles)

    def setattr(self, name):
        '''
        set the attribute of the layer class
        :param name: str
            name of attribute
        :return: None
        '''
        if name in self.kwargs:
            setattr(self, name, self.kwargs[name])

    def evaluate(self,input,dim,name,**op_dict):
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
        self.build(input,dim, name, **op_dict)

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

    def __init__(self, **kwargs):
        baselayer.__init__(self, **kwargs)

    def apply(self):
        if len(self.children) > 1:
            feedlist = []
            for child in self.children:
                chd = child
                input = self.input
                if isinstance(child, list):
                    chd = child[0]
                    input = child[1]
                feedlist.append(chd.feedforward(input))
            return concatenate(feedlist, axis=feedlist[0].ndim - 1)
        elif len(self.children) == 1:
            child = self.children.items()[0][1]
            chd = child
            input = self.input
            if isinstance(child, list):
                chd = child[0]
                input = child[1]
            return chd.feedforward(input)
        else:
            return self.input


class linear(layer):
    '''
    linear layer
    '''

    def __init__(self, unit, activation=None, **kwargs):
        layer.__init__(self, **kwargs)
        self.unit_dim = unit
        self.activation = activation

    def init_params(self):
        self.wt = self._allocate(uniform, 'Wt', weight, self.in_dim, self.unit_dim)

    def apply(self):
        if self.activation is not None:
            return self.activation(T.dot(self.input, self.wt))
        else:
            return T.dot(self.input, self.wt)

    def public_ops(self):
        pass


class linear_bias(linear):
    '''
    linear layer with bias
    '''

    def __init__(self, unit, activation=None, **kwargs):
        linear.__init__(self, unit, activation, **kwargs)

    def init_params(self):
        linear.init_params(self)
        self.bi = self._allocate(zeros, 'Bi', bias, self.unit_dim)

    def apply(self):
        if self.activation is not None:
            return self.activation(T.dot(self.input, self.wt) + self.bi)
        else:
            return T.dot(self.input, self.wt) + self.bi


class hidden_layer(layer):
    '''
    setup base hidden layer
    '''

    def __init__(self, unit, activation=T.tanh, **kwargs):
        layer.__init__(self, **kwargs)
        self.unit_dim=unit
        self.children['linear_bias'] = linear_bias(unit, activation)


class output_layer(layer):
    ''' 
    setup base output layer 
    '''

    def __init__(self, unit, activation=T.nnet.sigmoid, **kwargs):
        layer.__init__(self, **kwargs)
        self.unit_dim=unit
        self.children['linear_bias'] = linear_bias(unit, activation)
        self.cost_function = mean_square
        self.cost = None
        self.predict = None
        self.error = None
        self.setattr('cost_function')
        self.setattr('cost')
        self.setattr('predict')
        self.setattr('error')

    def get_output(self):
        layer.get_output(self)

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
