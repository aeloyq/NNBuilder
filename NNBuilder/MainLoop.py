# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:37:12 2016

@author: aeloyq
"""
#TODO:类化，装饰器加入，处理extension
import timeit
import theano
import numpy as np
import copy

def Train(configuration, model_stream, datastream,extension):
    datastream,train_model,valid_model,test_model,sample_model,debug_model,model,NNB_model,optimizer=model_stream
    train_X, valid_X, test_X, train_Y, valid_Y, test_Y = datastream
    print "\r\nTrainning Model:\r\n"
    train_minibatches, valid_minibatches, test_minibatches=get_minibatches_idx(configuration,datastream)
    sample_data=[datastream[0],datastream[3]]
    batch_size=configuration['batch_size']
    max_epoches=configuration['max_epoches']
    iteration_total=[0]
    epoch_time=[0.]
    train_result=[]
    debug_result=[]
    train_error=[1.]
    test_error=[1.]
    best_valid_error=[1.]
    best_iter=[0]
    epoches=[0]
    errors=[]
    costs=[]
    first=True
    dict=configuration
    dict['model_stream']=model_stream
    dict['iteration_total']=iteration_total
    dict['minibatches']=[train_minibatches,valid_minibatches,test_minibatches]
    dict['data_stream']=datastream
    dict['sample_data']=sample_data
    dict['epoch_time'] = epoch_time
    dict['train_error'] = train_error
    dict['train_result'] = train_result
    dict['test_error'] = test_error
    dict['debug_result'] = debug_result
    dict['best_valid_error'] = best_valid_error
    dict['best_iter'] = best_iter
    dict['epoches'] = epoches
    dict['errors'] = errors
    dict['costs'] = costs
    dict['first'] = first
    dict['stop']=False
    extension_instance=[]
    for ex in extension:ex.config.kwargs=dict;extension_instance.append(ex.config)
    # Main Loop
    print '        ','Training Start'
    for ex in extension_instance:   ex.before_train()
    if dict['stop']:
        return -1, [], [], debug_result
    while(True):
        # Train model iter by iter
        for _,index in train_minibatches:
            x = [train_X[0][t] for t in index]
            y = [train_Y[t] for t in index]
            data=[x,y]
            for idx,other_data in enumerate(train_X[1:]):
                o_d=[train_X[idx][t] for t in index]
                data.extend(o_d)
            data=tuple(data)
            train_result=train_model(*data)
            train_cost=train_result[0]
            iteration_total[0] += 1
            for ex in extension_instance:   ex.after_iteration()
            if dict['stop']:
                return epoches[0],errors,costs,debug_result
            # Stop When Sucess
            if train_error == 0:
                testdatas = []
                for _, index in test_minibatches:
                    x = [test_X[t] for t in index]
                    y = [test_Y[t] for t in index]
                    testdatas.append([x, y])
                if np.mean([test_model(*tuple(testdata)) for testdata in testdatas]) == 0:
                    best_iter = iteration_total
                    print "\r\n●Trainning Sucess●\r\n"
                    break
        # After epoch
        testdatas = []
        for _, index in test_minibatches:
            x = [test_X[t] for t in index]
            y = [test_Y[t] for t in index]
            testdatas.append([x, y])
        train_error[0] = np.mean([test_model(*tuple(testdata)) for testdata in testdatas])
        errors.append(train_error[0])
        costs.append(train_cost)
        epoches[0] += 1
        for ex in extension_instance:   ex.after_epoch()
        # Stop When Timeout
        if epoches[0] > max_epoches - 1 and max_epoches != -1:
            print "⊙Trainning Time Out⊙"
            break
    for ex in extension_instance:   ex.after_train()
    return epoches[0],errors,costs,debug_result

def get_minibatches_idx(conf,datastream, shuffle=False):
    train_X, valid_X, test_X, train_Y, valid_Y, test_Y = datastream
    train_X=train_X[0]
    minibatch_size=conf['batch_size']
    try:
        n_train_batches = train_X.get_value().shape[0]
        n_valid_batches = valid_X.get_value().shape[0]
        n_test_batches = test_X.get_value().shape[0]
    except:
        n_train_batches = len(train_X)
        n_valid_batches = len(valid_X)
        n_test_batches = len(test_X)
    def get_(n,minibatch_size,shuffle=None):
        idx_list = np.arange(n, dtype="int32")

        if shuffle:
            np.random.shuffle(idx_list)

        minibatches = []
        minibatch_start = 0
        for i in range(n // minibatch_size):
            minibatches.append(idx_list[minibatch_start:
                                        minibatch_start + minibatch_size])
            minibatch_start += minibatch_size

        if (minibatch_start != n):
            # Make a minibatch out of what is left
            minibatches.append(idx_list[minibatch_start:])

        return zip(range(len(minibatches)), minibatches)
    train_minibatches=get_(n_train_batches,minibatch_size)
    valid_minibatches = get_(n_valid_batches, minibatch_size)
    test_minibatches = get_(n_test_batches, minibatch_size)
    return train_minibatches,valid_minibatches,test_minibatches



