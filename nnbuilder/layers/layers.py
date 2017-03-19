# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:55:42 2016

@author: aeloyq
"""

from nnbuilder import config
import numpy as np
import theano
import theano.tensor as T


''' base class '''

class baselayer:
    def __init__(self):
        self.rng=config.rng
        self.name='unnamed'
        self.input=None
        self.output=None
        self.ops_on_output = []
        self.debug_stream=[]
        self.param_init_functions=paraminitfunctions()
        self.params=[]
        self.ops=None
    def add_debug(self,additem):
        self.debug_stream.append(additem)
    def set_input(self,X):
        self.input=X
    def get_output(self):
        self.output=self.input
    def build(self):
        for op in self.ops_on_output:
            self.output = op(self.output)
    def set_name(self,name):
        self.name=name

    def init_layer_params(self):
        pass

class paraminitfunctions:
    def __init__(self):
        self.uniform=layer_tools.uniform
        self.zeros=layer_tools.zeros
        self.randn=layer_tools.randn
        self.orthogonal=layer_tools.orthogonal

class costfunctions:
    def __init__(self):
        self.square = layer_tools.square_cost
        self.cross_entropy = layer_tools.cross_entropy_cost
        self.neglog = layer_tools.neglog_cost


class layer(baselayer):
    def __init__(self, in_dim, unit_dim, activation=None, **kwargs):
        baselayer.__init__(self)
        self.in_dim=in_dim
        self.unit_dim=unit_dim
        self.param_init_function={'wt':self.param_init_functions.uniform,'bi':self.param_init_functions.zeros}
        if 'name' in kwargs:
            self.name = kwargs['name']
        if 'param_init_function' in kwargs:
            self.param_init_function = kwargs['param_init_function']
        self.activation=activation
        self.wt='wt'
        self.bi='bi'
        self.params=[self.wt,self.bi]
    def init_layer_params(self):
        wt_values = self.param_init_function['wt'](self.in_dim,self.unit_dim)
        bi_values = self.param_init_function['bi'](self.unit_dim)
        self.wt = theano.shared(value=wt_values, name='Wt'+'_'+self.name, borrow=True)
        self.bi = theano.shared(value=bi_values, name='Bi'+'_'+self.name, borrow=True)
        self.params = [self.wt, self.bi]
    def load_weights(self,*args):
        self.wt=args[0]
        self.bi=args[1]
        self.params = [self.wt, self.bi]
    def set_weight_init_function(self,*args):
        self.param_init_function=args
    def get_output(self):
        if self.activation is not None:
            self.output=self.activation(T.dot(self.input,self.wt)+self.bi)
        else:
            self.output=T.dot(self.input,self.wt)+self.bi
    def set_activation(self,method):
        self.activation=method


''' setup base hidden layer '''

class hidden_layer(layer):
    def __init__(self,in_dim, unit_dim, activation=T.tanh,**kwargs):
        layer.__init__(self, in_dim, unit_dim, activation, **kwargs)
    
''' setup base output layer '''

class output_layer(layer):
    def __init__(self, in_dim, unit_dim,Activation=T.nnet.sigmoid,**kwargs):
        layer.__init__(self, in_dim, unit_dim, Activation, **kwargs)
        self.cost_functions=costfunctions()
        self.cost_function=self.cost_functions.square
    def get_output(self):

        if self.activation is not None:
            if self.input.ndim == 2:
                self.output=self.activation(T.dot(self.input,self.wt)+self.bi)
            elif self.input.ndim == 3:
                out, up = theano.scan(lambda i: self.activation(T.dot(i, self.wt) + self.bi),
                                      sequences=[self.input], n_steps=self.input.shape[0])
                self.output = out
        else:
            self.output=T.dot(self.input,self.wt)+self.bi

        self.predict()
    def predict(self):
        self.pred_Y=T.round(self.output)
    def cost(self,Y):
        if Y.ndim==2:
            return self.cost_function(Y, self.output)
        if Y.ndim==1:
            return self.cost_function(T.reshape(Y, [Y.shape[0], 1]),  self.output)
    def error(self,Y):
        if Y.ndim == 1:
            return layer_tools.errors(T.reshape(Y, [Y.shape[0], 1]), self.pred_Y)
        if Y.ndim==2:
            return layer_tools.errors(Y, self.pred_Y)

    def add_Y_mask(self,tvar):
        self.Y_mask=tvar

''' tools for building layers '''

class layer_tools:
    def __init__(self):
        pass    
    # weights init function

    @staticmethod
    def numpy_asarray_floatx(data):
        data2return=np.asarray(data,dtype=theano.config.floatX)
        return data2return

    @staticmethod
    def randn(*args):
        param=config.rng.randn(*args)
        return layer_tools.numpy_asarray_floatx(param)

    @staticmethod
    def uniform(*args):
        param = config.rng.uniform(low=-np.sqrt(6. / sum(args)),high=np.sqrt(6. / sum(args)),size=args)
        return layer_tools.numpy_asarray_floatx(param)

    @staticmethod
    def zeros(*args):
        shape=[]
        for dim in args:
            shape.append(dim)
        param = np.zeros(shape)
        return layer_tools.numpy_asarray_floatx(param)

    @staticmethod
    def orthogonal(*args):
        param = layer_tools.uniform(*args)
        param=np.linalg.svd(param)[0]
        return layer_tools.numpy_asarray_floatx(param)

    # recurrent output
    @staticmethod
    def final(outputs,mask):
        return outputs[-1]

    @staticmethod
    def all(outputs,mask):
        return outputs

    @staticmethod
    def mean_pooling(outputs,mask):
        return ((outputs * mask[:, :, None]).sum(axis=0))/mask.sum(axis=0)[:, None]


    # cost function
    @staticmethod
    def square_cost(Y_reshaped,outputs_reshaped):
        return T.sum(T.square(Y_reshaped -outputs_reshaped))/2
    @staticmethod
    def neglog_cost(Y_reshaped,outputs_reshaped):
        return -T.mean(T.log(outputs_reshaped)[T.arange(Y_reshaped.shape[0]),Y_reshaped])
    @staticmethod
    def cross_entropy_cost(Y_reshaped,outputs_reshaped):
        return -T.mean(Y_reshaped*T.log(outputs_reshaped)+(1-Y_reshaped)*T.log(1-outputs_reshaped))
    # calculate the error rates
    @staticmethod    
    def errors(Y_reshaped,pred_Y):
        return T.mean(T.neq(pred_Y, Y_reshaped))