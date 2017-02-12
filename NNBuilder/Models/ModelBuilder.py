# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:36:32 2016

@author: aeloyq
"""

import theano
import theano.tensor as T

''' Modeling MLP '''

def Model_Constructor(configuration, datastream):
    print "Building Model"
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
    NNB_model=configuration['model_output'](Rng=configuration['rng'], N_in=configuration['n_inputs'],
                                               N_out=configuration['n_outputs'],Raw_Input=X,
                                               l0=configuration['L0_reg'],
                                               l1=configuration['L1_reg'], l2=configuration['L2_reg'],
                                               Wt_init=configuration['Wt_init'],Bi_init=configuration['Bi_init']
                                               ,Hidden_Layer_Struct=configuration['model_struct'],Cost_func=configuration['cost_func'])
    cost = NNB_model.cost(Y)
    theano.pp(cost)
    params=[wt_bi for param in NNB_model.wt_packs for wt_bi in param]
    gparams = T.grad(cost, params)
    updates = [(param, param - configuration['learning_rate'] * gparam)
               for param, gparam in zip(params, gparams)]
    print 'Compiling Training Model'
    train_model = theano.function(inputs=[iteration_train],
                                  outputs=NNB_model.cost(Y),
                                  updates=updates,
                                  givens={X: train_X[
                                             (iteration_train - 1) * configuration['batch_size']:iteration_train *
                                                                                                 configuration[
                                                                                                     'batch_size']],
                                          Y: train_Y[
                                             (iteration_train - 1) * configuration['batch_size']:iteration_train *
                                                                                                 configuration[
                                                                                                     'batch_size']]})
    print 'Compiling Validing Model'
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
    print 'Compiling Test Model'
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
    print 'Compiling Sampling Model'
    sample_model = theano.function(inputs=[X,Y],
                                 outputs=[NNB_model.pred_Y,NNB_model.cost(Y),NNB_model.error(Y)])
    print 'Compiling Model'
    model = theano.function(inputs=[X],
                            outputs=NNB_model.pred_Y)
    return [train_model, valid_model, test_model, sample_model, model, NNB_model, n_train_batches, n_valid_batches,n_test_batches]
