# -*- coding: utf-8 -*-
"""
Created on  Feb 13 10:24 PM 2017

@author: aeloyq
"""
import Layers
import theano
import theano.tensor as T
import numpy as np

def Get_Model_Stream(configuration, datastream,algrithm,dim_model=None):
    print "\r\nBuilding Model:\r\n"
    train_X, valid_X, test_X, train_Y, valid_Y, test_Y = datastream
    try:
        n_train_batches = (train_X.get_value().shape[0] - 1) // configuration['batch_size'] + 1
        n_valid_batches = (valid_X.get_value().shape[0]-1) // configuration['batch_size']+1
        n_test_batches = (test_X.get_value().shape[0]-1) // configuration['batch_size']+1
    except:
        n_train_batches = (train_X.shape[0] - 1) // configuration['batch_size'] + 1
        n_valid_batches = (valid_X.shape[0] -1)// configuration['batch_size']+1
        n_test_batches = (test_X.shape[0]-1) // configuration['batch_size']+1
    index = T.lscalar('index')
    if dim_model is not None:
        NNB_model=dim_model
    else:
        dim_model=load_model(configuration)
        NNB_model=dim_model
    X=NNB_model.X
    Y=NNB_model.Y
    NNB_model.get_cost_pred_error()
    wt_bi = dim_model.wt_packs
    cost = NNB_model.cost
    error=NNB_model.error
    pred_Y=NNB_model.pred_Y
    main_algrithm=algrithm[0]
    algrithm_instance = main_algrithm.algrithm(configuration, wt_bi,cost)
    updates=algrithm_instance.get_updates(algrithm_instance)
    for other_algrithm in algrithm[1:]:
        pass
    debug_output = [X, Y, pred_Y, cost]
    print '        ', 'Compiling Training Model'
    train_model = theano.function(inputs=[index],
                                  outputs=cost,
                                  updates=updates,
                                  givens={X: train_X[
                                             index * configuration['batch_size']:(index + 1) *
                                                                                 configuration[
                                                                                     'batch_size']],
                                          Y: train_Y[
                                             index * configuration['batch_size']:(index + 1) *
                                                                                 configuration[
                                                                                     'batch_size']]})
    print '        ','Compiling Validing Model'
    valid_model = theano.function(inputs=[index],
                                  outputs=error,
                                  givens={X: valid_X[
                                             index * configuration['batch_size']:(index + 1) *
                                                                                           configuration[
                                                                                               'batch_size']],
                                          Y: valid_Y[
                                             index * configuration['batch_size']:(index + 1) *
                                                                                           configuration[
                                                                                               'batch_size']]})
    print '        ','Compiling Test Model'
    test_model = theano.function(inputs=[index],
                                 outputs=error,
                                 givens={X: test_X[
                                            index * configuration['batch_size']:(index + 1) *
                                                                                           configuration[
                                                                                               'batch_size']],
                                          Y: test_Y[
                                             index * configuration['batch_size']:(index + 1) *
                                                                                           configuration[
                                                                                               'batch_size']]})
    print '        ','Compiling Sampling Model'
    sample_model = theano.function(inputs=[X,Y],
                                 outputs=[pred_Y,cost,error])
    print '        ','Compiling Debug Model'
    debug_model = theano.function(inputs=[index],
                                  outputs=debug_output,
                                  givens={X: train_X[
                                             index* configuration['batch_size']:(index+1) *
                                                                                                 configuration[
                                                                                                     'batch_size']],
                                          Y: train_Y[
                                             index * configuration['batch_size']:(index+1) *
                                                                                                 configuration[
                                                                                                     'batch_size']]})
    print '        ','Compiling Model'
    model = theano.function(inputs=[X],
                            outputs=pred_Y)
    return [train_model, valid_model, test_model, sample_model, debug_model,model, NNB_model, n_train_batches, n_valid_batches,n_test_batches,wt_bi,cost]

class model():
    def __init__(self,configurations):
        self.X = T.matrix('X')
        self.Y = T.ivector('Y')
        self.rng=configurations['rng']
        self.l0 = configurations['L0_reg']
        self.l1 = configurations['L1_reg']
        self.l2 = configurations['L2_reg']
        self.layer_stream={}
        self.outputs={'raw':self.X}
        self.final_output=None
        self.layer_dict = {}
        self.wt_packs=[]
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

    def set_final_output(self,name):
        self.final_output=self.layer_stream[name]
        self.fn_output=self.final_output.outputs

    def get_cost_pred_error(self):
        self.pred_Y=self.final_output.pred_Y
        self.cost=self.final_output.cost(self.Y)+self.regularization(self.wt_packs,self.l0,self.l1,self.l2)
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

