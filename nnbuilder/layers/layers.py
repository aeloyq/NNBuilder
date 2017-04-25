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

class ops:
    class dropout:
        name='dropout'
        use_noise='use_noise'
        def __init__(self):
            self.name='dropout'
        @staticmethod
        def op(tvar,**kwargs):
            return tvar*config.trng.binomial(tvar.shape,
                          p=kwargs['use_noise'], n=1,
                          dtype=tvar.dtype)
        @staticmethod
        def op_(tvar,**kwargs):
            return tvar*(1-kwargs['use_noise'])
    class residual:
        name='residual'
        def __init__(self):
            self.name='residual'
        @staticmethod
        def op(tvar,**kwargs):
            return tvar+kwargs['pre_tvar']
        @staticmethod
        def op_(tvar, **kwargs):
            return ops.residual.op(tvar,**kwargs)
    class batch_normalization:
        name='batch_normalization'
        def __init__(self):
            self.name='batch_normalization'
        @staticmethod
        def op(tvar,**kwargs):
            return T.nnet.batch_normalization(tvar,kwargs['gamma'],kwargs['beta'],kwargs['mean'],kwargs['std'])
        @staticmethod
        def op_(tvar, **kwargs):
            return ops.batch_normalization.op(tvar, **kwargs)

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
            >>> x, y = T.matrices('x', 'y')
            >>> c = utils.concatenate([x, y], axis=1)
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

        out = T.zeros(output_shape)
        offset = 0
        for tt in tensor_list:
            indices = ()
            for k in range(axis):
                indices += (slice(None),)
            indices += (slice(offset, offset + tt.shape[axis]),)
            for k in range(axis + 1, tensor_list[0].ndim):
                indices += (slice(None),)

            out = T.set_subtensor(out[indices], tt)
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
    '''
     base class of layer
    '''
    def __init__(self,**kwargs):
        '''
        initiate the layer class to an instance
        :param kwargs: 
        '''
        self.kwargs=kwargs
        self.rng=config.rng
        self.name='None'
        self.input=None
        self.output=None
        self.children=OrderedDict()
        self.params=OrderedDict()
        self.roles=OrderedDict()
        self.updates=OrderedDict()
        self.ops=OrderedDict()
        self.debug_stream=[]
        self.setattr('rng')
        self.setattr('name')
        self.setattr('input')
        self.setattr('output')
        self.setattr('children')
        self.setattr('params')
        self.setattr('roles')
        self.setattr('updates')
        self.setattr('ops')
        self.setattr('debug_stream')
    def set_name(self,name):
        '''
        set the name of layer
        :param name: str
            name of layer
        :return: None
        '''
        self.name=name
    def set_input(self,X):
        '''
        set the input of layer
        :param X: tensor variable
            input tensor
        :return: None
        '''
        self.input=X
    def get_output(self):
        '''
        get the output of layer
        :return: tensor variable
            output of layer
        '''
        self.apply()
        self.public_ops()
    def public_ops(self):
        '''
        public ops on tensor variable
        such like dropout batch normalization etc.
        if want to change the sequence of ops
        please overwrite this function
        :return: None
        '''
        self.output=self.addops('output',self.output,ops.batch_normalization)
        self.output=self.addops('output',self.output,ops.dropout)
        self.output=self.addops('output',self.output,ops.residual)
    def addops(self,name,tvar,ops,switch=True):
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
        name=name+'_'+ops.name+'_'+self.name
        if name not in self.ops:self.ops[name]=switch
        if not self.ops[name]:return tvar
        if name in self.op_dict:
            dict=self.op_dict[name]
        else:
            dict=self.op_dict[ops.name]
        if self.op_dict['mode']=='train':
            return ops.op(tvar,**dict)
        if self.op_dict['mode']=='use':
            return ops.op_(tvar,**dict)
    def apply(self):
        '''
        build the graph of layer
        :return: tensor variable
            the computational graph of this layer
        '''
        return self.input
    def _allocate(self,role,name,rndfn,*args):
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
        '''
        initiate the params
        :return: None
        '''
        pass
    def build(self,X,name,**op_dict):
        '''
        build the layer
        :param X: tensor variable
            input of layer
        :param name: 
            name of layer
        :return: tensor variable
            output of layer
        '''
        self.op_dict=op_dict
        self.set_name(name)
        self.init_params()
        self.set_input(X)
        self.merge()
        self.get_output()
        return self.output
    def merge(self):
        '''
        merge the children to the layer
        :return: None
        '''
        for name,child in self.children:
            inp=self.input
            chd=child
            if isinstance(child,list):
                chd=child[0]
                inp=child[1]
            chd.build(inp,self.name+'_'+name,self.op_dict)
            self.params.update(child.params)
            self.roles.update(child.roles)
    def setattr(self,name):
        '''
        set the attribute of the layer class
        :param name: str
            name of attribute
        :return: None
        '''
        if name in self.kwargs:
            setattr(self,name,self.kwargs[name])
    def add_debug(self,*additem):
        '''
        add tensor variable to debug stream
        :param additem: tensor variable
            tensor variable which want to be debug in debug mode
        :return: None
        '''
        self.debug_stream.extend(list(additem))



class layer(baselayer):
    '''
    abstract layer
    '''
    def __init__(self,**kwargs):
        baselayer.__init__(self,**kwargs)
    def apply(self):
        if self.children != []:
            return utils.concatenate([child.output for child in self.children],axis=self.children[0].output.ndim-1)
        else:
            return self.input


class linear(layer):
    '''
    linear layer
    '''
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
    '''
    linear layer with bias
    '''
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
    '''
    setup base hidden layer
    '''
    def __init__(self,in_dim, unit_dim, activation=T.tanh,**kwargs):
        layer.__init__(self, **kwargs)
        self.children['lb']=linear_bias(in_dim, unit_dim, activation)

class output_layer(layer):
    ''' 
    setup base output layer 
    '''
    def __init__(self, in_dim, unit_dim,activation=T.nnet.sigmoid,**kwargs):
        layer.__init__(self, **kwargs)
        self.children['lb'] = linear_bias(in_dim, unit_dim, activation)
        self.cost_func=utils.square_cost
        self.cost=None
        self.predict = None
        self.error = None
        self.setattr('cost_func')
        self.setattr('cost')
        self.setattr('predict')
        self.setattr('error')
    def get_output(self):
        layer.get_output(self)
        self.predict()
    def get_predict(self):
        '''
        get the predict of the model
        :return: tensor variable
            the predict of model
        '''
        self.predict=T.round(self.output)
    def get_cost(self,Y):
        '''
        get the cost of the model
        :param Y: 
            the label of the model which used to evaluate the cost function(loss function)
        :return: tensor variable
            the cost of the model
        '''
        self.cost=self.cost_func(Y,self.output)
    def get_error(self,Y):
        '''
        get the error of the model
        :param Y: 
            the label of the model which used to caculate the error
        :return: tensor variable
            the error (1-accruate%) of the model
        '''
        self.error=T.mean(T.neq(Y,self.predict))





