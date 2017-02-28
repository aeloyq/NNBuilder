# -*- coding: utf-8 -*-
"""
Created on  Feb 13 10:24 PM 2017

@author: aeloyq
"""
import Layers
import theano
import theano.tensor as T
import numpy as np

def Get_Model_Stream(configuration, datastream,algrithm,dim_model=None,grad_monitor=False):
    print "\r\nBuilding Model:\r\n"
    index = T.lscalar('index')
    if dim_model is not None:
        configuration['load_model']=dim_model
        NNB_model=load_model(configuration)
    else:
        dim_model=load_model(configuration)
        NNB_model=dim_model
    X=NNB_model.X
    Y=NNB_model.Y
    train_inputs=NNB_model.train_inputs
    other_inputs=NNB_model.other_inputs
    NNB_model.get_cost_pred_error()
    wt_bi = NNB_model.wt_packs
    cost = NNB_model.cost
    error=NNB_model.error
    pred_Y=NNB_model.pred_Y
    optimizer = algrithm.config
    optimizer.init(wt_bi,cost)
    train_updates=optimizer.get_updates()
    debug_output=[]
    for layer in NNB_model.layer_stream:
        for sublayer in NNB_model.layer_stream[layer]:
            debug_output.extend(sublayer.debug_stream)
    train_output=[cost]
    if grad_monitor:
        train_output.extend(optimizer.gparams)
    print '        ', 'Compiling Training Model'
    train_model = theano.function(inputs=train_inputs,
                                  outputs=train_output,
                                  updates=train_updates)
    print '        ','Compiling Validing Model'
    valid_model = theano.function(inputs=other_inputs,
                                  outputs=error)
    print '        ','Compiling Test Model'
    test_model = theano.function(inputs=other_inputs,
                                 outputs=error)
    print '        ','Compiling Sampling Model'
    sample_model = theano.function(inputs=other_inputs,
                                 outputs=[pred_Y,cost,error])
    print '        ','Compiling Debug Model'
    debug_model = theano.function(inputs=train_inputs,
                                  outputs=debug_output,
                                  on_unused_input='ignore')
    print '        ','Compiling Model'
    model = theano.function(inputs=[X],
                            outputs=pred_Y)
    return [datastream,train_model, valid_model, test_model, sample_model, debug_model,model, NNB_model,optimizer]






class model():
    def __init__(self,configurations,in_dims=2):
        if in_dims==2:
            self.X = T.matrix('X')
        elif in_dims==3:
            self.X = T.tensor3('X')
        self.Y = T.ivector('Y')
        self.train_inputs=[self.X,self.Y]
        self.other_inputs=[self.X,self.Y]
        self.rng=configurations['rng']
        self.layer_stream={}
        self.outputs={'raw':self.X}
        self.final_output=None
        self.layer_dict = {}
        self.wt_packs=[]
        self.scan_updates=None
        layers = Layers.__all__
        for ly in layers:
            if ly!='Layers':
                self.layer_dict[ly] = eval('Layers.' + ly + '.layer')

    def addlayer(self,layer_info,inputs,name,num=1):
        layer2add=[]
        input_stream=[self.outputs[inputs]]
        if isinstance(layer_info,list):
            layer=layer_info[0]
            conf=layer_info[1]
        else:
            layer = layer_info
            conf=None
        for n in range(num):
            if layer in self.layer_dict:
                layer2add.append(self.layer_dict[layer](self.rng,*conf))
            else:
                layer2add.append(layer)
            layer2add[-1].set_name(name)
            layer2add[-1].init_wt_bi()
            layer2add[-1].set_inputs(input_stream[-1])
            input_stream.append(layer2add[-1].outputs)
            self.wt_packs.append(layer2add[-1].params)
        self.layer_stream[name] = layer2add
        #if layer2add[-1].outputs
        self.outputs[name] = layer2add[-1].outputs
        self.final_output=layer2add[-1]

    def addpointwise(self,operation,layer1,layer2,name):
        opt_output=None
        if operation == '+':
            opt_output = layer1.output + layer2.output
        elif operation == '-':
            opt_output = layer1.output - layer2.output
        elif operation == '*':
            opt_output = layer1.output * layer2.output
        elif operation == '&':
            opt_output = T.concatenate([layer1.output, layer2.output, 1])
        self.outputs[name] = opt_output

    def adddropout(self,operation,layer1,layer2):
        opt_output=None
        if operation == '+':
            opt_output = layer1.output + layer2.output
        elif operation == '-':
            opt_output = layer1.output - layer2.output
        elif operation == '*':
            opt_output = layer1.output * layer2.output
        elif operation == '&':
            opt_output = T.concatenate([layer1.output, layer2.output, 1])

    def adddropout2all(self,operation,layer1,layer2):
        opt_output=None
        if operation == '+':
            opt_output = layer1.output + layer2.output
        elif operation == '-':
            opt_output = layer1.output - layer2.output
        elif operation == '*':
            opt_output = layer1.output * layer2.output
        elif operation == '&':
            opt_output = T.concatenate([layer1.output, layer2.output, 1])

    def adddropout2ff(self,operation,layer1,layer2):
        opt_output=None
        if operation == '+':
            opt_output = layer1.output + layer2.output
        elif operation == '-':
            opt_output = layer1.output - layer2.output
        elif operation == '*':
            opt_output = layer1.output * layer2.output
        elif operation == '&':
            opt_output = T.concatenate([layer1.output, layer2.output, 1])

    def adddropout2recurrent(self,operation,layer1,layer2):
        opt_output=None
        if operation == '+':
            opt_output = layer1.output + layer2.output
        elif operation == '-':
            opt_output = layer1.output - layer2.output
        elif operation == '*':
            opt_output = layer1.output * layer2.output
        elif operation == '&':
            opt_output = T.concatenate([layer1.output, layer2.output, 1])

    def add_theano_input(self,variable):
        self.train_inputs.append(variable)

    def set_final_output(self,name):
        self.final_output=self.layer_stream[name]
        self.fn_output=self.final_output.outputs

    def get_cost_pred_error(self):
        self.pred_Y=self.final_output.pred_Y
        self.cost=self.final_output.cost(self.Y)#+self.regularization(self.wt_packs,0,0,0.0001)
        self.error=self.final_output.error(self.Y)
    def regularization(self,wt_packs,l0,l1,l2):
        reg0=reg1=reg2=0
        for wt_bi in wt_packs:
            for wt in wt_bi[:-1]:
                if l0 != 0:
                    if reg0!=0:
                        reg0 =reg0+T.nonzero_values(wt)
                    else:
                        reg0 = T.nonzero_values(wt)
                if l1 != 0:
                    if reg1 != 0:
                        reg1=reg1+abs(wt).sum()
                    else:
                        reg1 = abs(wt).sum()
                if l2 != 0:
                    if reg2 != 0:
                        reg2=reg2+(wt**2).sum()
                    else:
                        reg2 = (wt ** 2).sum()
        reg=l0*reg0+l1*reg1+l2*reg2
        return reg


def load_model(configurations):
    model2return=model(configurations)
    for i in configurations['load_model']:
        if i[1]=='layer':
            model2return.addlayer(*i[0])
        elif i[1]=='pointwise':
            model2return.addpointwise(*i[0])
    return model2return

