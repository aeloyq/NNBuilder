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
from collections import OrderedDict
import logger

class model():
    def __init__(self):
        self.X = T.matrix('X')
        self.Y = T.ivector('Y')
        self.inputs=[self.X,self.Y]
        self.train_inputs=[self.X,self.Y]
        self.model_inputs=[self.X,self.Y]
        self.rng=config.rng
        self.layers=OrderedDict()
        self.nodes=[]
        self.output=None
        self.cost=None
        self.error=None
        self.graph=[]
        self.params=[]
        self.ops_on_cost=[]
        self.user_debug_stream=[]
        self.trng=config.trng
        self.X_mask=None
        self.Y_mask=None

    def set_inputs(self, inputs):
        self.inputs=inputs
        self.train_inputs=inputs
        self.X=inputs[0]
        self.Y = inputs[1]


    def set_output(self,layer):
        self.output=layer.output

    def build(self):
        for node in self.graph:
            node.evaulate()
        self.output=self.graph[-1].layer
        for key in self.layers:
            self.user_debug_stream.extend(self.layers[key].debug_stream)

    def get_cost_pred_error(self):
        self.pred_Y=self.output.pred_Y
        self.cost=self.output.cost(self.Y)
        for ops in self.ops_on_cost:
            self.cost=ops.evaulate(self.cost)
        self.error=self.output.error(self.Y)

    def addlayer(self,layer,input,name=None):
        layer_instance=self.layer(layer,input,name)
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
            try:
                self.layer.set_input(self.input.output)
            except:
                self.layer.set_input(self.input)
            self.layer.get_output()
            self.output=self.layer.output

    def addpointwise(self,operation,layer1,layer2):
        logger("Warning!:this is deprecated in this version please try to use softplus instead", 0)
        pointwise_instance=self.pointwise(operation,layer1,layer2)
        self.nodes.append(pointwise_instance)
        self.graph.append(pointwise_instance)
        return pointwise_instance

    class pointwise:
        def __init__(self,operation,id1=None,id2=None):
            self.operation=operation
            self.id1=id1
            self.id2=id2
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
                                                        p=self.use_noise, n=1,
                                                        dtype=self.layer.output.dtype)),
                                    self.layer.output * (1-self.use_noise))


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
                for param in  layers[key].params:
                    if param.name.find('Bi')==-1:
                        self.params.append(param)
        def evaulate(self,cost):
            reg = 0
            for param in self.params:
                reg+=T.sum(param**2)
            return cost+self.l2*reg

    def add_residual(self, layer,pre_layer):
        residual_instance = self.weight_decay(layer, pre_layer)
        self.ops_on_cost.append(residual_instance)

    class residual:
        def __init__(self, layer, pre_layer):
            self.layer = layer
            self.pre_layer = pre_layer
            self.output = None

        def evaulate(self):
            self.output = self.layer.output+self.pre_layer.output




