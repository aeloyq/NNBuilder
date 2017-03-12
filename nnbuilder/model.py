# -*- coding: utf-8 -*-
"""
Created on  Feb 13 10:24 PM 2017

@author: aeloyq
"""
import config
import layers
import theano
import theano.tensor as T
import numpy as np


class model():
    def __init__(self):
        self.X = T.matrix('X')
        self.Y = T.ivector('Y')
        self.inputs=[self.X,self.Y]
        self.train_inputs=[self.X,self.Y]
        self.model_inputs=[self.X,self.Y]
        self.rng=config.rng
        self.layers={}
        self.nodes={}
        self.output=None
        self.cost=None
        self.error=None
        self.graph=[]
        self.params=[]
        self.ops_on_cost=[]
        self.trng=config.trng

    def set_inputs(self,train_inputs,model_inputs):
        self.inputs=train_inputs
        for input in model_inputs:
            if input not in self.inputs:
                self.inputs.append(input)
        self.train_inputs=train_inputs
        self.model_inputs=model_inputs


    def set_output(self,layer):
        self.output=layer.output

    def build(self):
        for node in self.graph:
            node.evaulate()
        self.output=self.graph[-1].layer

    def get_cost_pred_error(self):
        self.pred_Y=self.output.pred_Y
        self.cost=self.output.cost(self.Y)
        for ops in self.ops_on_cost:
            self.cost=ops.evaulate(self.cost)
        self.error=self.output.error(self.Y)

    def addlayer(self,layer,input,name=None,mask=None):
        layer_instance=self.layer(layer,input,name,mask)
        self.layers[name]=(layer)
        self.graph.append(layer_instance)
        self.params.append(layer.params)

    class layer():
        def __init__(self,layer,input,name=None,mask=None):
            self.layer=layer
            self.input=input
            self.name=name
            self.mask=mask
            if name is not None: layer.set_name(name)
            self.layer.init_layer_params()
        def evaulate(self):
            if self.mask is not None: self.layer.set_mask = self.mask
            try:
                self.layer.set_input(self.input.output)
            except:
                self.layer.set_input(self.input)
            self.layer.get_output()
            self.output=self.layer.output

    def addpointwise(self,operation,layer1,layer2,name):
        pw=self.pointwise(operation,layer1,layer2,name)
        self.nodes[name]=pw
        self.graph.append(pw)

    class pointwise:
        def __init__(self,operation,id1=None,id2=None,name=None):
            self.operation=operation
            self.id1=id1
            self.id2=id2
            self.name=name
            self.output=None
        def evaulate(self):
            if self.operation == '+':
                self.output = self.id1.output + self.id2.output
            elif self.operation == '-':
                self.output = self.id1.output - self.id2.output
            elif self.operation == '*':
                self.output = self.id1.output * self.id2.output
            elif self.operation == '&':
                self.output = T.concatenate([self.id1.output, self.id2.output, 1])


    def add_dropout(self,layer,use_noise=0.5):
        if use_noise == 0.5:
            try:
                use_noise=config.use_noise
            except:
                pass
        drop_out_instance=self.dropout(layer,self.trng,'dropout_%s'%layer.name,use_noise)
        self.graph.append(drop_out_instance)

    class dropout:
        def __init__(self,layer,trng,name=None,use_noise=0.5):
            self.layer=layer
            self.use_noise=theano.shared(value=use_noise,name='dropout_noise%s'%layer.name,borrow=True)
            self.output=None
            self.trng=trng
            self.name=name
        def evaulate(self):
            self.output=self.layer.output = T.switch(self.use_noise,
                                    (self.layer.output *
                                     self.trng.binomial(self.layer.output.shape,
                                                        p=0.5, n=1,
                                                        dtype=self.layer.output.dtype)),
                                    self.layer.output * 0.5)


    def add_weight_decay(self,l2=0.0001, layers=None):
        if layers==None:layers=self.layers #TODO: may cause incorrect result
        dropout_instance=self.weight_decay(layers,l2)
        self.ops_on_cost.append(dropout_instance)


    class weight_decay():
        def __init__(self,layers,l2=0.0001):
            self.layers=layers
            self.l2=l2
            self.params=[]
            for key in layers:
                self.params.extend(layers[key].params[:-1])
        def evaulate(self,cost):
            reg = 0
            for param in self.params:
                reg+=T.sum(param**2)
            return cost+self.l2*reg



