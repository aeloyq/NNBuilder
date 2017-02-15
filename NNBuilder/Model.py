# -*- coding: utf-8 -*-
"""
Created on  Feb 13 10:24 PM 2017

@author: aeloyq
"""
import Layers
import theano
import theano.tensor as T

def Get_Model_Stream(configuration, datastream,dim_model=None):
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
    X = T.matrix('X')
    Y = T.ivector('Y')
    iteration_train = T.lscalar('iteration_train')
    iteration_valid = T.lscalar('iteration_valid')
    iteration_test = T.lscalar('iteration_test')
    if dim_model is not None:
        NNB_model=dim_model.final_output
        wt_bi=dim_model.wt_packs
    else:
        dim_model=load_model(configuration,X)
        NNB_model=dim_model.final_output
        wt_bi = dim_model.wt_packs
    cost = NNB_model.cost(Y,wt_bi)
    params=[theta for param in wt_bi for theta in param]
    gparams = T.grad(cost, params)
    outputs=[cost]
    for gparam in gparams:
        outputs.append(gparam)
    print '        ','Compiling Training Model'
    train_model = theano.function(inputs=[iteration_train],
                                  outputs=outputs,
                                  givens={X: train_X[
                                             (iteration_train - 1) * configuration['batch_size']:iteration_train *
                                                                                                 configuration[
                                                                                                     'batch_size']],
                                          Y: train_Y[
                                             (iteration_train - 1) * configuration['batch_size']:iteration_train *
                                                                                                 configuration[
                                                                                                     'batch_size']]})
    print '        ','Compiling Validing Model'
    valid_model = theano.function(inputs=[iteration_valid],
                                  outputs=NNB_model.error(Y),
                                  givens={X: valid_X[
                                             (iteration_valid - 1) * configuration['batch_size']:iteration_valid *
                                                                                                 configuration[
                                                                                                     'batch_size']],
                                          Y: valid_Y[
                                             (iteration_valid - 1) * configuration['batch_size']:iteration_valid *
                                                                                                 configuration[
                                                                                                     'batch_size']]})
    print '        ','Compiling Test Model'
    test_model = theano.function(inputs=[iteration_test],
                                 outputs=NNB_model.error(Y),
                                 givens={X: test_X[
                                             (iteration_test - 1) * configuration['batch_size']:iteration_test *
                                                                                                 configuration[
                                                                                                     'batch_size']],
                                         Y: test_Y[
                                             (iteration_test - 1) * configuration['batch_size']:iteration_test *
                                                                                                 configuration[
                                                                                                     'batch_size']]})
    print '        ','Compiling Sampling Model'
    sample_model = theano.function(inputs=[X,Y],
                                 outputs=[NNB_model.pred_Y,cost,NNB_model.error(Y)])
    print '        ','Compiling Debug Model'
    debug_model = theano.function(inputs=[iteration_train],
                                  outputs=[X,Y,NNB_model.outputs,NNB_model.pred_Y,cost,gparams[0]],
                                  givens={X: train_X[
                                             (iteration_train - 1) * configuration['batch_size']:iteration_train *
                                                                                                 configuration[
                                                                                                     'batch_size']],
                                          Y: train_Y[
                                             (iteration_train - 1) * configuration['batch_size']:iteration_train *
                                                                                                 configuration[
                                                                                                     'batch_size']]})
    print '        ','Compiling Model'
    model = theano.function(inputs=[X],
                            outputs=NNB_model.pred_Y)
    return [train_model, valid_model, test_model, sample_model, debug_model,model, NNB_model, n_train_batches, n_valid_batches,n_test_batches,wt_bi,cost]

def load_model(configurations,raw_inputs):
    model2return=model(configurations,raw_inputs)
    for i in configurations['load_model']:
        if i[1]=='layer':
            model2return.addlayer(*i[0])
        elif i[1]=='pointwise':
            model2return.addpointwise(*i[0])
    return model2return

class model():
    def __init__(self,configurations,raw_inputs):
        self.rng=configurations['rng']
        self.layer_stream={}
        self.outputs={'raw':raw_inputs}
        self.final_output=None
        self.layer_dict = {}
        self.wt_packs=[]
        layers = Layers.__all__
        for ly in layers:
            if ly!='Layers':
                self.layer_dict[ly] = eval('Layers.' + ly + '.layer')

    def addlayer(self,layer,layer_conf,inputs,name,num=1):
        if layer in self.layer_dict:
            layer2add=[]
            input_stream=[self.outputs[inputs]]
            conf=layer_conf
            for n in range(num):
                layer2add.append(self.layer_dict[layer](self.rng,*conf))
                conf=list(conf)
                conf[0]=conf[1]
                conf=tuple(conf)
                layer2add[-1].set_inputs(input_stream[-1])
                input_stream.append(layer2add[-1].outputs)
                self.wt_packs.append(layer2add[-1].params)
            self.layer_stream[name] = layer2add
            self.outputs[name] = layer2add[-1].outputs
            self.final_output=layer2add[-1]

    def addpointwise(self,operation,layer1,layer2,name):
        if operation == '+':
            output = layer1.output + layer2.output
        elif operation == '-':
            output = layer1.output - layer2.output
        elif operation == '*':
            output = layer1.output * layer2.output
        elif operation == '&':
            output = T.concatenate([layer1.output, layer2.output, 1])
        self.outputs[name] = output

    def set_final_output(self,name):
        self.final_output=self.layer_stream[name]

