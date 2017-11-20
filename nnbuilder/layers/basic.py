# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:55:42 2016

@author: aeloyq
"""

import copy
from ops import *
from param import *
from utils import *
from nnbuilder.kernel import *
from collections import OrderedDict


class LayerBase(object):
    def __init__(self, **kwargs):
        '''
        The Base Class of Layer
        :param kwargs:
            extra configurations
        :attr  name: str
            name of Layer
        :attr  in_dim: int
            input dimentionality of Layer
        :attr  out_dim: int
            output dimentionality of Layer
        :attr  root: Layer
            root Layer of current Layer
        :attr  children: Layer
            children Layer of current Layer
        :attr  input: kernel.tensor
            input tensor of Layer
        :attr  output: kernel.tensor
            output tensor of Layer when training
        :attr  running_output: kernel.tensor
            output tensor of Layer when running(predicting)
        :attr  outputs: kernel.tensor
            output tensors of Layer when training
        :attr  running_outputs: kernel.tensor
            output tensors of Layer when running(predicting)
        :attr  params: OrderedDict
            parameters of Layer
        :attr  updates: OrderedDict
            updates dictionary of tensor of Layer when training
        :attr  running_updates: OrderedDict
            updates dictionary of tensor of Layer when running(predicting)
        :attr  ops: Ops
            Ops class which contains added training trick classes of Layer
        :attr  debug_stream: list
            tensors need to be debugged in Layer
        '''
        self.name = None
        self.in_dim = None
        self.out_dim = None
        self.root = self
        self.children = OrderedDict()
        self.input = None
        self.output = None
        self.outputs = OrderedDict()
        self.updates = OrderedDict()
        self.params = OrderedDict()
        self.running_output = None
        self.running_outputs = OrderedDict()
        self.running_updates = OrderedDict()
        self.ops = Ops()
        self.debug_stream = []
        self._model_inputs = OrderedDict()
        self._model_outputs = OrderedDict()
        self._build_mode = None
        self._ops_option = OrderedDict()
        self._pre_layer = None
        self._next_layer = None
        self._kwargs = kwargs
        self.setattr('name')
        self.setattr('in_dim')
        self.setattr('input')

    def __str__(self):
        return self.name

    def __setattr__(self, key, value):
        if key in ['root', '_pre_layer', '_next_layer']:
            self.__dict__[key] = value
        elif isinstance(value, LayerBase):
            if not hasattr(self, key):
                self.__dict__[key] = value
                self.children[key] = value
                value.set_root(self.root)
        elif isinstance(value, Parameter):
            if not hasattr(self, key):
                self.__dict__[key] = value
                self.params[value.name] = value
        elif isinstance(value, Op):
            if not hasattr(self, key):
                self.__dict__[key] = value
                self.ops.update(value)
                value.name = key
                value.update({'name': key})
        else:
            self.__dict__[key] = value

    def setattr(self, name, attr=None):
        '''
        set an attribute of Layer
        :param name(str):
            name of attribute of Layer
        :param attr(any): None
            attribute of Layer
        :return: None
        '''
        if name in self._kwargs:
            setattr(self, name, self._kwargs[name])
        elif attr is not None:
            setattr(self, name, attr)

    def set_name(self, name):
        '''
        set the name of layer
        :param name(str):
            set name of Layer
        :return: None
        '''
        self.name = name

    def set_pre_layer(self, pre_layer):
        '''
        set previous Layer of current Layer
        :param pre_layer(Layer):
            previous Layer
        :return: None
        '''
        self._pre_layer = pre_layer

    def set_root(self, root):
        '''
        set root Layer of current Layer
        :param root(Layer):
             root Layer of current Layer
        :return: None
        '''
        self.root = root

    def set_in_dim(self, dim):
        '''
        set input dimentionality of Layer
        :param dim(int):
            input dimentionality of Layer
        :return: None
        '''
        if self.in_dim is None:
            if isinstance(dim, tuple):
                dim = list(dim)
            self.in_dim = dim

    def get_out_dim(self):
        '''
        set output dimentionality of Layer
        :return: None
        '''
        self.out_dim = self.output.size[-1]

    def set_input(self, X):
        '''
        set input tensor of Layer
        :param X(kernel.tensor):
            nput tensor of Layer
        :return: None
        '''
        self.input = X

    def set_ops(self):
        '''

        :return: None
        '''
        pass

    def set_model_inputs(self, model_inputs):
        '''
        set inputs of Model
        :param model_input(OrderedDict):
            inputs of Model
        :return: None
        '''
        self._model_inputs = model_inputs

    def set_model_outputs(self, model_outputs):
        '''
        set outputs of Model
        :param model_input(OrderedDict):
            inputs of Model
        :return: None
        '''
        self._model_outputs = model_outputs

    def get_units(self):
        '''

        :return:
        '''
        units = [self]
        for child in self.children:
            units.extend(child.get_units())
        return units

    def show_units(self):
        '''

        :return:
        '''
        units = [self.name]
        for child in self.children:
            units.extend(child.show_units())
        return units

    def get_ops(self):
        '''

        :return:
        '''
        ops = self.ops.get()
        for child in self.children:
            ops.extend(child.get_ops())
        return ops

    def show_ops(self):
        '''

        :return:
        '''
        ops = self.ops.show()
        for child in self.children:
            ops.extend(child.show_ops())
        return ops

    def update_ops(self, op_instance):
        '''

        :param op_instance:
        :param units:
        :return:
        '''
        op2update = []
        if op_instance.op_units is None:
            units = self.get_units()
            for unit in units:
                ops = unit.ops.get()
                for op in ops:
                    if isinstance(op_instance, op.op):
                        op2update.append(op)
        else:
            for op in op_instance.op_units:
                if isinstance(op_instance, op):
                    op2update.append(op)
        for op in op2update:
            op.update(op_instance.options)
            op.switch = True

    def set_children(self):
        '''
        set the children of Layer
        :return: None
        '''
        pass

    def set_params(self):
        '''
        set the parameters of Layer
        :return: None
        '''
        pass

    def init_params(self):
        '''
        initiate the parameters of Layer
        :return: None
        '''
        for param in self.params.values():
            param.init()

    def apply(self, X):
        '''
        kernel function of Layer
        build your main layer structure here
        get the output graph of layer given X(input)
        :param X: kernel.tensor
            input graph of Layer
        :return: kernel.tensor or list or tuple
            output graphs of layer given X
        '''
        return X

    def feed(self, X):
        '''
        feed forward to get the output graph of Layer
        the wrapper of apply
        if you have multiple outputs yield from apply()
        then you have to change this function
        :return: kernel.tensor
            the output graph of Layer
        '''
        outputs = OrderedDict()
        outputs['output'] = self.apply(X)
        return outputs, OrderedDict()

    def sample(self, output):
        '''
        get the sample results of Layer given output
        :param output(kernel.tensor):
             sample results of Layer given output
        :return: OrderedDict
            must have a key named 'predict' which refers to the prediction of Layer
        '''
        sample = OrderedDict()
        sample['predict'] = T.round(output)
        return sample

    def predict(self, output):
        '''
        get the prediction of Layer given output
        :param output(kernel.tensor):
             prediction of Layer given output
        :return: OrderedDict
            must have a key named 'predict' which refers to the prediction of Layer
        '''
        return self.sample(output)

    def build_prepare(self):
        self.set_in_dim(self.root._pre_layer.out_dim)
        self.set_children()
        for name, child in self.children.items():
            child.initiate(self.name + '_' + name, self._pre_layer, self._model_inputs, self._model_outputs)
            child.build_prepare()
        self.set_params()
        self.init_params()

    def build_train(self):
        '''
        build Layer in training mode
        :return: None
        '''
        self._build_mode = 'train'
        self.set_input(self.root._pre_layer.output)
        self.outputs, self.updates = self.feed(self.input)
        self.output = self.outputs['output']
        self.merge_train()
        self.get_out_dim()
        return self.output

    def build_running(self):
        '''
        build Layer in running mode
        :return: None
        '''
        self._build_mode = 'running'
        self.set_input(self._pre_layer.running_output)
        self.running_outputs, self.running_updates = self.feed(self.input)
        self.running_output = self.running_outputs['output']
        self.merge_running()
        return self.running_output

    def merge_train(self):
        '''
        merge children of Layer
        :return: None
        '''
        for name, child in self.children.items():
            for name, param in child.params.items():
                self.params[name] = param
            self.updates.update(child.updates)

    def merge_running(self):
        '''
        merge children of Layer
        :return: None
        '''
        for name, child in self.children.items():
            self.running_updates.update(child.running_updates)

    def initiate(self, name, prelayer, model_inputs, model_outputs):
        '''
        initiate Layer and merge children of Layer
        :param name(str):
            name of Layer
        :param prelayer(Layer):
            previous Layer of current Layer
        :param model_input(OrderedDict):
            input of Model
        :return: None
        '''
        self.set_name(name)
        self.set_pre_layer(prelayer)
        self.set_model_inputs(model_inputs)
        self.set_model_outputs(model_outputs)

    def debug(self, *variable):
        '''
        add tensor variable to debug stream of Layer
        :param variable: kernel.tensor
            tensor variable which want to be debug in debug mode
        :return: None
        '''
        self.debug_stream.extend(list(variable))
