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

''' setup base hidden layer '''

class Hidden_Layer(Layer):
    def __init__(self,Rng,N_in,N_hl,Wt=None,Bi=None,Wt_init='uniform',Bi_init='zeros',Activation=T.nnet.sigmoid):
        Layer.__init__(self,Rng,N_in,N_hl,Wt,Bi,Wt_init,Bi_init,Activation)
    
''' setup base output layer '''

class Output_Layer(Layer):
    Hidden_Layer_Struct=[];Cost_func='square'
    def __init__(self,Rng,N_in,N_out,Raw_Input=None,l0=0.,l1=0.,l2=0.,Wt=None,Bi=None,Wt_init='uniform',Bi_init='zeros',Hidden_Layer_Struct=[],Cost_func='square',Activation=T.nnet.sigmoid):
        Layer.__init__(self,Rng,N_in,N_out,Wt,Bi,Wt_init,Bi_init,Activation)        
        self.Raw_Input=Raw_Input
        self.l0=l0
        self.l1=l1
        self.l2=l2
        self.Hidden_Layer_Struct=Hidden_Layer_Struct
        self.Cost_func=Cost_func
        self.calc()
        self.predict()
    def calc(self):
        self.outputs,self.wt_packs,self.model=Layer_Tools.builder(self.Raw_Input,self.Hidden_Layer_Struct,[self.params,self.Activation])
    def output_func(self):
        pass
    def predict(self):
        self.pred_Y=T.round(self.outputs)
    def cost(self,Y):
        return Layer_Tools.cost(T.reshape(Y,[Y.shape[0],1]),self.Cost_func,self.outputs,self.wt_packs,self.l0,self.l1,self.l2)
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
        return cost_dict[cost_func](Y,outputs,wt_packs,l0,l1,l2)
    @staticmethod
    def square_cost(Y_reshaped,outputs_reshaped,wt_packs,l0,l1,l2):
        return T.sum(T.square(Y_reshaped -outputs_reshaped))/2+Layer_Tools.regularization(wt_packs,l0,l1,l2)
    @staticmethod
    def neglog_cost(Y_reshaped,outputs_reshaped,wt_packs,l0,l1,l2):
        return -1*T.mean(T.log(outputs_reshaped))+Layer_Tools.regularization(wt_packs,l0,l1,l2)
    @staticmethod
    def cross_entropy_cost(Y_reshaped,outputs_reshaped,wt_packs,l0,l1,l2):
        return -T.mean(Y_reshaped*T.log(outputs_reshaped)+(1-Y_reshaped)*T.log(1-outputs_reshaped))+Layer_Tools.regularization(wt_packs,l0,l1,l2)
    # calculate the error rates
    @staticmethod    
    def errors(Y_reshaped,pred_Y):
        return T.mean(T.neq(pred_Y, Y_reshaped))
    # decode the hidden and output layer struct pack and trans them into a model
    @staticmethod    
    def builder(Raw_Input,Hidden_Layer_Struct,Output_Layer_param):
        operation_count=index_count=0
        input_dict={'raw':Raw_Input}
        model=[]
        last_outputs=None
        wt_pack=[]
        builder_dict=['+','*','&'] 
        for unit in Hidden_Layer_Struct:
            if (unit not in builder_dict):
                model.append(unit[0][0](*unit[0][1]))
                model[-1].set_inputs(input_dict[unit[2]])
                wt_pack.append(model[-1].params)
                input_dict[unit[1]]=model[-1].outputs
                index_pointer=index_count
                last_outputs=model[-1].outputs
                unit_to_operate=unit
                for i in range(operation_count):
                    index_pointer+=1
                    n=unit_to_operate[3]-1
                    unit_to_operate=Hidden_Layer_Struct[index_pointer]
                    operation_flag=Hidden_Layer_Struct[index_count-operation_count]
                    for j in n:
                        model.append(unit_to_operate[0][0](*unit_to_operate[0][1]))
                        model[-1].set_inputs(input_dict[unit_to_operate[2]])
                        wt_pack.append(model[-1].params)
                    input_dict[unit_to_operate[1]]=model[-1].outputs
                    if operation_flag[0]=='+':
                        last_outputs=last_outputs+model[-1].outputs
                    elif operation_flag[0]=='-':
                        last_outputs=last_outputs-model[-1].outputs
                    elif operation_flag[0]=='*':
                        last_outputs=last_outputs*model[-1].outputs
                    elif operation_flag[0]=='&':
                        last_outputs=T.concatenate([last_outputs,model[-1].outputs,1])
                    input_dict[operation_flag[1]]=last_outputs
                    operation_count-=1
            elif unit in builder_dict:
                operation_count+=1
            index_count+=1
        last_outputs=T.dot(last_outputs,Output_Layer_param[0][0])+Output_Layer_param[0][1]
        if Output_Layer_param[1] is not None:
            last_outputs=Output_Layer_param[1](last_outputs)
        wt_pack.append(Output_Layer_param[0])
        return last_outputs,wt_pack,model
                





















class MLP_Softmax(object):
    def __init__(self,Rng,inputs,N_in,N_hl,N_out,l0=0.,l1=0.,l2=0.,Wt=None,Bi=None,Hidden_Activation=T.nnet.sigmoid):
        self.inputs=inputs
        self.l0=l0
        self.l1=l1
        self.l2=l2
        #self.hidden=Hidden_Layer_FeedForward(Rng,inputs,N_in,N_hl,Activation=Hidden_Activation)
        if Wt is None:
            Wt_values=np.array(Rng.randn(N_hl,N_out),dtype=theano.config.floatX)
            Wt=theano.shared(value=Wt_values,name='Wt',borrow=True)
        if Bi is None:
            Bi_values=np.zeros((N_out), dtype=theano.config.floatX)
            Bi = theano.shared(value=Bi_values, name='Bi', borrow=True)
        self.Wt,self.Bi=Wt,Bi
        self.outputs=T.nnet.nnet.softmax(T.dot(self.hidden.outputs,self.Wt)+self.Bi)
        self.pred_Y=T.argmax(self.outputs, axis=1)
        self.params=self.hidden.params+[self.Wt,self.Bi]
    
    def square_cost(self,Y):
        return T.mean(T.square(np.array(1,dtype='float32')-self.outputs[T.arange(Y.shape[0]),Y]))/2+self.regularization()
    def neglog_cost(self,Y):
        return T.mean(-T.log(T.max(self.outputs,1)))/2+self.regularization()
    def cross_entropy_cost(self,Y):
        return T.mean(T.square(T.reshape(Y,[Y.shape[0],1]) -T.max(self.outputs,1)))/2+self.regularization()
    def errors(self,Y):
        return T.mean(T.neq(self.pred_Y, Y))
        
def regularization(self):
        reg0=reg1=reg2=0
        if reg0!=0:reg0=np.count_nonzero(self.hidden.Wt)+np.count_nonzero(self.Wt)
        if reg1!=0:reg1+=(abs(self.hidden.Wt)).sum()+(abs(self.Wt)).sum()
        if reg2!=0:reg2+=(self.hidden.Wt**2).sum()+(self.Wt**2).sum()
        reg=reg0*self.l0+T.mean(reg1)*self.l1+T.mean(reg2)*self.l2
        return reg
'''def Cost_Calculator(outputs,Y):
    if square_cost(self,Y):
        return T.mean(T.square(np.array(1,dtype='float32')-self.outputs[T.arange(Y.shape[0]),Y]))/2+self.regularization()
    def neglog_cost(self,Y):
        return T.mean(-T.log(T.max(self.outputs,1)))/2+self.regularization()
    def cross_entropy_cost(self,Y):
        return T.mean(T.square(T.reshape(Y,[Y.shape[0],1]) -T.max(self.outputs,1)))/2+self.regularization()
    def errors(self,Y):
        return T.mean(T.neq(self.pred_Y, Y))'''
        