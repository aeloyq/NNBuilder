# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:55:42 2016

@author: aeloyq
"""

from nnbuilder import config
from collections import OrderedDict
import numpy as np
import theano
import theano.tensor as T


class roles:
    class weight:
        pass
    class bias:
        pass
weight=roles.weight
bias=roles.weight

class utils:
    ''' tools for building layers '''
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
        return utils.numpy_asarray_floatx(param)

    @staticmethod
    def uniform(*args):
        param = config.rng.uniform(low=-np.sqrt(6. / sum(args)),high=np.sqrt(6. / sum(args)),size=args)
        return utils.numpy_asarray_floatx(param)

    @staticmethod
    def zeros(*args):
        shape=[]
        for dim in args:
            shape.append(dim)
        param = np.zeros(shape)
        return utils.numpy_asarray_floatx(param)

    @staticmethod
    def orthogonal(*args):
        param = utils.uniform(args[0], args[0])
        param = np.linalg.svd(param)[0]
        for _ in range(args[1]/args[0]-1):
            param_ = utils.uniform(args[0], args[0])
            param=np.concatenate((param,np.linalg.svd(param_)[0]),1)
        return utils.numpy_asarray_floatx(param)

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

    @staticmethod
    def concatenate(tensor_list, axis=0):
        """
        Alternative implementation of `theano.tensor.concatenate`.
        This function does exactly the same thing, but contrary to Theano's own
        implementation, the gradient is implemented on the GPU.
        Backpropagating through `theano.tensor.concatenate` yields slowdowns
        because the inverse operation (splitting) needs to be done on the CPU.
        This implementation does not have that problem.
        :usage:
            >>> x, y = theano.tensor.matrices('x', 'y')
            >>> c = concatenate([x, y], axis=1)
        :parameters:
            - tensor_list : list
                list of Theano tensor expressions that should be concatenated.
            - axis : int
                the tensors will be joined along this axis.
        :returns:
            - out : tensor
                the concatenated tensor expression.
        """
        concat_size = sum(tt.shape[axis] for tt in tensor_list)

        output_shape = ()
        for k in range(axis):
            output_shape += (tensor_list[0].shape[k],)
        output_shape += (concat_size,)
        for k in range(axis + 1, tensor_list[0].ndim):
            output_shape += (tensor_list[0].shape[k],)

        out = tensor.zeros(output_shape)
        offset = 0
        for tt in tensor_list:
            indices = ()
            for k in range(axis):
                indices += (slice(None),)
            indices += (slice(offset, offset + tt.shape[axis]),)
            for k in range(axis + 1, tensor_list[0].ndim):
                indices += (slice(None),)

            out = tensor.set_subtensor(out[indices], tt)
            offset += tt.shape[axis]

        return out


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

uniform=utils.uniform
zeros=utils.zeros



class baselayer:
    ''' base class '''
    def __init__(self,**kwargs):
        self.kwargs=kwargs
        self.setattr('rng',config.rng)
        self.setattr('name','None')
        self.setattr('input')
        self.setattr('output')
        self.setattr('children',OrderedDict())
        self.setattr('params', OrderedDict())
        self.setattr('roles', OrderedDict())
        self.setattr('updates', OrderedDict())
        self.setattr('debug_stream', OrderedDict())
    def set_name(self,name):
        self.name=name
    def set_input(self,X):
        self.input=X
    def get_output(self):
        self.apply()
    def apply(self):
        return self.input
    def _allocate(self,role,name,rndfn,*args):
        if name in self.kwargs:
            if callable(self.kwargs[name]):rndfn=self.kwargs[name]
            else:
                self.params[name] = theano.shared(value=self.kwargs[name], name=name + '_' + self.name, borrow=True)
                self.roles[name] = role
                return self.params[name]
        if role is roles.weight:
            self.params[name] = theano.shared(value=rndfn(*args), name=name + '_' + self.name, borrow=True)
            self.roles[name]=role
        return self.params[name]

    def init_params(self):
        pass
    def build(self,X,name):
        self.set_name(name)
        self.init_params()
        self.set_input(X)
        self.merge()
        self.get_output()
    def merge(self):
        for name,child in self.children:
            inp=self.input
            chd=child
            if isinstance(child,list):
                chd=child[0]
                inp=child[1]
            chd.build(inp,self.name+'_'+name)
            self.params.update(child.params)
            self.roles.update(child.roles)
    def setattr(self,name,value=None):
        if name in self.kwargs:
            setattr(self,name,self.kwargs[name])
        else:
            setattr(self, name, value)
    def add_debug(self,*additem):
        self.debug_stream.extend(list(additem))



class layer(baselayer):
    def __init__(self,**kwargs):
        baselayer.__init__(self,**kwargs)
    def apply(self):
        if self.children != []:
            return utils.concatenate([child.output for child in self.children],axis=self.children[0].output.ndim-1)
        else:
            return self.input


class linear(layer):
    def __init__(self, in_dim, unit_dim, activation=None,**kwargs):
        layer.__init__(self,**kwargs)
        self.in_dim=in_dim
        self.unit_dim=unit_dim
        self.activation=activation
    def init_params(self):
        self.wt = self._allocate(uniform,'Wt',weight,self.in_dim,self.unit_dim)
    def apply(self):
        if self.activation is not None:
            self.output=self.activation(T.dot(self.input,self.wt))
        else:
            self.output=T.dot(self.input,self.wt)

class linear_bias(linear):
    def __init__(self, in_dim, unit_dim, activation=None, **kwargs):
        linear.__init__(self, in_dim, unit_dim, activation, **kwargs)
    def init_params(self):
        linear.init_params(self)
        self.bi = self._allocate(uniform,'Bi',weight,self.unit_dim)
    def apply(self):
        if self.activation is not None:
            self.output=self.activation(T.dot(self.input,self.wt)+self.bi)
        else:
            self.output=T.dot(self.input,self.wt)+self.bi


class hidden_layer(layer):
    ''' setup base hidden layer '''
    def __init__(self,in_dim, unit_dim, activation=T.tanh,**kwargs):
        layer.__init__(self, **kwargs)
        self.children['lb']=linear_bias(in_dim, unit_dim, activation)

class output_layer(layer):
    ''' setup base output layer '''
    def __init__(self, in_dim, unit_dim,Activation=T.nnet.sigmoid,**kwargs):
        layer.__init__(self, **kwargs)
        self.cost_func=utils.square_cost
        self.cost=None
        self.predict = None
        self.error = None
        if 'cost_func' in kwargs:
            self.cost_func = kwargs['cost_func']
    def get_output(self):
        layer.get_output()
        self.predict()
    def get_predict(self):
        self.predict=T.round(self.output)
    def get_cost(self,Y):
        self.cost=self.cost_fn(Y,self.output)
    def get_error(self,Y):
        self.error=T.mean(T.neq(Y,self.predict))





