# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:55:42 2016

@author: aeloyq
"""

import numpy as np
import theano
import theano.tensor as T

''' base class '''

class baselayer:
    def __init__(self):
        self.debug_stream=[]
        pass
    def add_debug(self,additem):
        self.debug_stream.append(additem)

class Layer(baselayer):
    Rng=None;N_in=0;N_units=0;name='';Wt=None;Bi=None;Wt_init='randn';Bi_init='zeros';Activation=None;Inputs=[]
    def __init__(self,Rng,N_in,N_units,Name='undefined',Wt=None,Bi=None,Wt_init='zeros',Bi_init='zeros',Activation=None):
        baselayer.__init__(self)
        self.Rng=Rng
        self.N_in=N_in
        self.N_units=N_units
        self.Name = Name
        self.Wt=Wt
        self.Bi=Bi
        self.Wt_init=Wt_init
        self.Bi_init=Bi_init
        self.Activation=Activation
        self.wt_bi_inited=False
    def init_wt_bi(self):
        if not self.wt_bi_inited:
            Wt_values,Bi_values=Layer_Tools.Fully_connected_weights_init(self.Rng,self.N_in,self.N_units,self.Wt,self.Bi,self.Wt_init,self.Bi_init)
            Wt=theano.shared(value=Wt_values,name='Wt'+'_'+self.Name,borrow=True)
            Bi = theano.shared(value=Bi_values, name='Bi'+'_'+self.Name, borrow=True)
            self.Wt,self.Bi=Wt,Bi
            self.wt_bi_pack()
            self.wt_bi_inited=True
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
        self.Name=name

''' setup base hidden layer '''

class Hidden_Layer(Layer):
    def __init__(self,Rng,N_in,N_hl,Name='undefined',Wt=None,Bi=None,Wt_init='randn',Bi_init='zeros',Activation=T.tanh):
        Layer.__init__(self,Rng,N_in,N_hl,Name,Wt,Bi,Wt_init,Bi_init,Activation)
    
''' setup base output layer '''

class Output_Layer(Layer):
    Hidden_Layer_Struct=[];Cost_func='square'
    def __init__(self,Rng,N_in,N_out,Name='undefined',Wt=None,Bi=None,Wt_init='randn',Bi_init='zeros',Cost_func='square',Activation=T.nnet.sigmoid):
        Layer.__init__(self,Rng,N_in,N_out,Name,Wt,Bi,Wt_init,Bi_init,Activation)
        self.Cost_func=Cost_func
    def output_func(self):
        Layer.output_func(self)
        self.predict()
    def predict(self):
        self.pred_Y=T.round(self.outputs)
    def cost(self,Y):
        self.add_debug(Y)
        self.add_debug(self.outputs)
        if Y.ndim==2:
            return Layer_Tools.cost(Y,self.Cost_func,self.outputs)
        if Y.ndim==1:
            return Layer_Tools.cost(T.reshape(Y,[Y.shape[0],1]),self.Cost_func,self.outputs)
    def error(self,Y):
        if Y.ndim == 1:
            return Layer_Tools.errors(T.reshape(Y,[Y.shape[0],1]),self.pred_Y)
        if Y.ndim==2:
            return Layer_Tools.errors(Y, self.pred_Y)

''' tools for building layers '''

class Layer_Tools:
    def __init__(self):
        pass    
    # weights init function
    @staticmethod    
    def Fully_connected_weights_init(Rng,N_in,N_units,Wt,Bi,Wt_init,Bi_init):
        wt=None;bi=None
        init_func={'uniform':Rng.uniform,'zeros':np.zeros,'randn':Rng.randn}
        if Wt is None:
            if Wt_init == 'zeros':
                wt=init_func[Wt_init]([N_in,N_units])
            elif Wt_init == 'uniform':
                wt=init_func[Wt_init](low=-np.sqrt(6. / (N_in + N_units)),high=np.sqrt(6. / (N_in + N_units)),size=(N_in,N_units))
            elif Wt_init == 'randn':
                wt=init_func[Wt_init](N_in,N_units)
        else:
            wt=Wt
        if Bi is None:
            if Bi_init is 'zeros':
                bi=init_func[Bi_init]((N_units,))
            elif Bi_init is 'uniform':
                bi=init_func[Bi_init](low=-np.sqrt(6. / (N_in + N_units)),high=np.sqrt(6. / (N_in + N_units)),size=(N_in,))
            elif Bi_init is 'randn':
                bi=init_func[Bi_init](N_units,)
        else:
            bi=Bi
        wt = np.asarray(wt,dtype=theano.config.floatX)
        bi = np.asarray(bi,dtype=theano.config.floatX)
        return wt,bi

    @staticmethod
    def Fully_connected_U_init(Rng,N_units,U,U_init):
        u=None
        init_func={'uniform':Rng.uniform,'zeros':np.zeros,'randn':Rng.randn}
        if U is None:
            if U_init == 'zeros':
                u=init_func[U_init]([N_units,N_units])
            elif U_init == 'uniform':
                u=init_func[U_init](low=-np.sqrt(6. / (N_units + N_units)),high=np.sqrt(6. / (N_units + N_units)),size=(N_units,N_units))
            elif U_init == 'randn':
                u=init_func[U_init](N_units,N_units)
        else:
            u=U
        u = np.asarray(u,dtype=theano.config.floatX)
        return u

    @staticmethod
    def Fully_connected_lstm_U_init(Rng, N_units, U, U_init):
        u = None
        init_func = {'uniform': Rng.uniform, 'zeros': np.zeros, 'randn': Rng.randn}
        if U is None:
            if U_init == 'zeros':
                u = init_func[U_init]([N_units, N_units])
            elif U_init == 'uniform':
                u = init_func[U_init](low=-np.sqrt(6. / (N_units + N_units)), high=np.sqrt(6. / (N_units + N_units)),
                                      size=(N_units, N_units*4))
            elif U_init == 'randn':
                u = init_func[U_init](N_units, N_units)
        else:
            u = U
        u = np.asarray(u, dtype=theano.config.floatX)
        return u

    @staticmethod
    def Fully_connected_emb_init(Rng,N_in,N_units,Wemb,Wemb_init):
        emb=None
        init_func={'uniform':Rng.uniform,'zeros':np.zeros,'randn':Rng.randn}
        if Wemb is None:
            if Wemb_init == 'zeros':
                emb=init_func[Wemb_init]([N_in,N_units])
            elif Wemb_init == 'uniform':
                emb=init_func[Wemb_init](low=-np.sqrt(6. / (N_in + N_units)),high=np.sqrt(6. / (N_in + N_units)),size=(N_in,N_units))
            elif Wemb_init == 'randn':
                emb=init_func[Wemb_init](N_in,N_units)
        else:
            emb=Wemb
        emb = np.asarray(emb,dtype=theano.config.floatX)
        return emb
    # cost function
    @staticmethod
    def cost(Y,cost_func,outputs):
        cost_dict={'square':Layer_Tools.square_cost,'neglog':Layer_Tools.neglog_cost,'cross_entropy':Layer_Tools.cross_entropy_cost}
        return cost_dict[cost_func](Y,outputs)
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