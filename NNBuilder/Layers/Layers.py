# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:55:42 2016

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T

''' base class '''

class Layer:
    Rng=None;N_in=0;N_units=0;Wt=None;Bi=None;Wt_init='uniform';Bi_init='zeros';Activation=None;Inputs=[]
    def __init__(self,Rng,N_in,N_units,Wt=None,Bi=None,Wt_init='zeros',Bi_init='zeros',Activation=None):
        self.Rng=Rng
        self.N_in=N_in
        self.N_units=N_units
        self.Wt=Wt
        self.Bi=Bi
        self.Wt_init=Wt_init
        self.Bi_init=Bi_init
        self.Activation=Activation
        self.init_wt_bi()
        self.wt_bi_pack()
    def init_wt_bi(self):
        Wt_values,Bi_values=Layer_Tools.Fully_connected_weights_init(self.Rng,self.N_in,self.N_units,self.Wt,self.Bi,self.Wt_init,self.Bi_init)
        Wt=theano.shared(value=Wt_values,name='Wt',borrow=True)
        Bi = theano.shared(value=Bi_values, name='Bi', borrow=True)
        self.Wt,self.Bi=Wt,Bi
    def wt_bi_pack(self):
        self.params=[self.Wt,self.Bi]
    def output_func(self):
        if self.Activation is not None:
            self.outputs=self.Activation(T.dot(self.Inputs,self.Wt)+self.Bi)
        else:
            self.outputs=T.dot(self.Inputs,self.Wt)+self.Bi
    def set_inputs(self,Inputs_X):
        self.Inputs=Inputs_X
        self.output_func()
    def set_name(self,name):
        self.name=name

''' setup base hidden layer '''

class Hidden_Layer(Layer):
    def __init__(self,Rng,N_in,N_hl,Wt=None,Bi=None,Wt_init='uniform',Bi_init='zeros',Activation=T.nnet.sigmoid):
        Layer.__init__(self,Rng,N_in,N_hl,Wt,Bi,Wt_init,Bi_init,Activation)
    
''' setup base output layer '''

class Output_Layer(Layer):
    Hidden_Layer_Struct=[];Cost_func='square'
    def __init__(self,Rng,N_in,N_out,l0=0.,l1=0.,l2=0.,Wt=None,Bi=None,Wt_init='uniform',Bi_init='zeros',Hidden_Layer_Struct=[],Cost_func='square',Activation=T.nnet.sigmoid):
        Layer.__init__(self,Rng,N_in,N_out,Wt,Bi,Wt_init,Bi_init,Activation)
        self.l0=l0
        self.l1=l1
        self.l2=l2
        self.Hidden_Layer_Struct=Hidden_Layer_Struct
        self.Cost_func=Cost_func
    def output_func(self):
        Layer.output_func(self)
        self.predict()
    def predict(self):
        self.pred_Y=T.round(self.outputs)
    def cost(self,Y,wt_packs):
        return Layer_Tools.cost(T.reshape(Y,[Y.shape[0],1]),self.Cost_func,self.outputs,wt_packs,self.l0,self.l1,self.l2)
    def error(self,Y):
        return Layer_Tools.errors(T.reshape(Y,[Y.shape[0],1]),self.pred_Y)

''' tools for building layers '''

class Layer_Tools:
    def __init__(self):
        pass    
    # weights init function
    @staticmethod    
    def Fully_connected_weights_init(Rng,N_in,N_units,Wt,Bi,Wt_init,Bi_init):
        wt=None;bi=None
        init_func={'uniform':Rng.uniform,'zeros':np.zeros}
        if Wt is None:
            if Wt_init == 'zeros':
                wt=init_func[Wt_init]([N_in,N_units])
            elif Wt_init == 'uniform':
                wt=init_func[Wt_init](-1,1,[N_in,N_units])
            elif Wt_init == 'randn':
                wt=init_func[Wt_init](N_in,N_units)
        if Bi is None:
            if Bi_init is 'zeros':
                bi=init_func[Bi_init]([N_units])
            elif Bi_init is 'uniform':
                bi=init_func[Bi_init](-1,1,[N_units])
            elif Bi_init is 'randn':
                bi=init_func[Bi_init](N_units)
        wt=np.array(wt,dtype=theano.config.floatX)
        bi=np.array(bi,dtype=theano.config.floatX)
        return wt,bi
    # regularization function
    @staticmethod 
    def regularization(wt_packs,l0,l1,l2):
        reg0=reg1=reg2=0
        #flat=lambda t: [_ for y in t for _ in flat(y)] if isinstance(t, Iterable) else [t]
        #wt_flattened=flat(wt_packs)
        for wt_flattened in wt_packs[:-1]:
            for wt in wt_flattened:
                if l0 != 0:
                    reg0=np.count_nonzero(wt)+np.count_nonzero(wt)
                if l1 != 0:
                    reg1+=wt.sum()
                if l2 != 0:
                    reg2+=(wt**2).sum()
        reg=reg0*l0+T.mean(reg1)*l1+T.mean(reg2)*l2
        return reg
    # cost function
    @staticmethod
    def cost(Y,cost_func,outputs,wt_packs,l0,l1,l2):
        cost_dict={'square':Layer_Tools.square_cost,'neglog':Layer_Tools.neglog_cost,'cross_entropy':Layer_Tools.cross_entropy_cost}
        return cost_dict[cost_func](Y,outputs)+Layer_Tools.regularization(wt_packs,l0,l1,l2)
    @staticmethod
    def square_cost(Y_reshaped,outputs_reshaped):
        return T.sum(T.square(Y_reshaped -outputs_reshaped))/2
    @staticmethod
    def neglog_cost(Y_reshaped,outputs_reshaped):
        return -T.mean(T.log(outputs_reshaped))
    @staticmethod
    def cross_entropy_cost(Y_reshaped,outputs_reshaped):
        return -T.mean(Y_reshaped*T.log(outputs_reshaped)+(1-Y_reshaped)*T.log(1-outputs_reshaped))
    # calculate the error rates
    @staticmethod    
    def errors(Y_reshaped,pred_Y):
        return T.mean(T.neq(pred_Y, Y_reshaped))