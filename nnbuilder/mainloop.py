# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:37:12 2016

@author: aeloyq
"""
#TODO:类化，装饰器加入，处理extension
import timeit
import theano
import theano.tensor as T
import numpy as np
import copy
import config
from logger import logger
import os
from extensions import debugmode

def train(datastream, model, algrithm, extension):
    if not os.path.exists('./%s' % config.name):
        os.mkdir('./%s' % config.name)
    if not os.path.exists('./%s/log' % config.name):
        os.mkdir('./%s/log' % config.name)
    if not os.path.exists('./%s/save' % config.name):
        os.mkdir('./%s/save' % config.name)
    if not os.path.exists('./%s/tmp' % config.name):
        os.mkdir('./%s/tmp' % config.name)
    model.build()
    dim_model=model
    print_config(model, algrithm, extension)
    if debugmode in extension:
        train_model,valid_model,test_model,sample_model,model,NNB_model,optimizer=get_modelstream(model,algrithm,False)
    else:
        train_model, valid_model, test_model, sample_model, model, NNB_model, optimizer = get_modelstream(model,algrithm)
    train_X, valid_X, test_X, train_Y, valid_Y, test_Y = datastream
    logger("Trainning Model:",0,1)
    train_minibatches, valid_minibatches, test_minibatches=get_minibatches_idx(datastream)
    sample_data=[datastream[0],datastream[3]]
    max_epoches=config.max_epoches
    dict_param={}
    dict_param['dim_model']=dim_model
    dict_param['algrithm'] = algrithm
    dict_param['logger'] = logger
    dict_param['prepare_data']=prepare_data
    dict_param['get_sample_data']=get_sample_data
    dict_param['conf'] = config
    dict_param['batch_size']=config.batch_size
    dict_param['train_model']=train_model
    dict_param['valid_model'] = valid_model
    dict_param['test_model'] = test_model
    dict_param['sample_model'] = sample_model
    dict_param['model']=model
    dict_param['iteration_total']=0
    dict_param['minibatches']=[train_minibatches,valid_minibatches,test_minibatches]
    dict_param['data_stream']=datastream
    dict_param['sample_data']=sample_data
    dict_param['train_error'] = 1
    dict_param['train_result'] = 1
    dict_param['test_error'] = 1
    dict_param['debug_result'] = []
    dict_param['best_valid_error'] = 1
    dict_param['best_iter'] = -1
    dict_param['epoches'] = 0
    dict_param['errors'] = []
    dict_param['costs'] = []
    dict_param['stop']=False
    extension_instance=[]
    for ex in extension:ex.config.kwargs=dict_param;ex.config.init();extension_instance.append(ex.config)
    dict_param['extension']=extension_instance
    # Main Loop
    logger('Training Start',1)
    for ex in extension_instance:   ex.before_train()
    if dict_param['stop']:
        return -1, [], [], dict_param['debug_result'],[train_model,valid_model,test_model,sample_model,model,NNB_model,optimizer]
    while(True):
        # Stop When Timeout
        if dict_param['epoches'] > max_epoches - 1 and max_epoches != -1:
            logger("⊙Trainning Time Out⊙", 1, 1)
            break
        # Train model iter by iter
        for idx,index in train_minibatches:
            data = prepare_data(train_X, train_Y, index)
            train_result=train_model(*data)
            dict_param['train_result'] = train_result[0]
            train_cost=train_result[0]
            dict_param['iteration_total'] += 1
            if(idx==train_minibatches[-1][0]):
                # After epoch
                dict_param['epoches'] += 1
                testdatas = []
                for _, index in test_minibatches:
                    data = prepare_data(test_X, test_Y, index)
                    testdatas.append(data)
                dict_param['train_error'] = np.mean([test_model(*tuple(testdata)) for testdata in testdatas])
                dict_param['errors'].append(dict_param['train_error'])
                dict_param['costs'].append(train_cost)
                for ex in extension_instance:   ex.after_epoch()
            for ex in extension_instance:   ex.after_iteration()
            if dict_param['stop']:
                for ex in extension_instance:   ex.after_train()
                return dict_param['epoches'],dict_param['errors'],dict_param['costs'],dict_param['debug_result'],[train_model,valid_model,test_model,sample_model,model,NNB_model,optimizer]
            # Stop When Sucess
            if dict_param['train_error'] == 0:
                testdatas = []
                for _, index in test_minibatches:
                    data = prepare_data(test_X,test_Y, index)
                    testdatas.append(data)
                if np.mean([test_model(*tuple(testdata)) for testdata in testdatas]) == 0:
                    dict_param['best_iter'] = dict_param['iteration_total']
                    logger( "●Trainning Sucess●",1,1)
                    break


    for ex in extension_instance:   ex.after_train()
    return dict_param['epoches'],dict_param['errors'],dict_param['costs'],dict_param['debug_result'],[train_model,valid_model,test_model,sample_model,model,NNB_model,optimizer]








def prepare_data(data_x,data_y,index):
    if not config.transpose_x:
        x = [data_x[t] for t in index]
    else:
        x = [data_x[t] for t in index]
        maxlen = max([len(d) for d in x])
        x = np.array(x)
        mask_x = np.ones([len(index), maxlen]).astype(theano.config.floatX)
        for idx, i in enumerate(x):
            for j in range(len(i), maxlen):
                i.append(np.zeros_like(i[0]).tolist())
                mask_x[idx, j] = 0
        x_new = []
        for idx in range(len(x[0])):
            x_new.append([x[i][idx] for i in range(len(x))])
        x = x_new
        mask_x = mask_x.transpose()

    if not config.transpose_y:
        y = [data_y[t] for t in index]
    else:
        y = [data_y[t] for t in index]
        maxlen = max([len(d) for d in y])
        y = np.array(y)
        mask_y = np.ones([len(index), maxlen]).astype(theano.config.floatX)
        for idx, i in enumerate(y):
            for j in range(len(i), maxlen):
                i.append(np.zeros_like(i[0]).tolist())
                mask_y[idx, j] = 0
        y_new = []
        for idx in range(len(y[0])):
            y_new.append([y[i][idx] for i in range(len(y))])
        y = y_new
        mask_y = mask_y.transpose()
    if config.int_x:x=np.asarray(x).astype('int64').tolist()
    if config.int_y:y=np.asarray(y).astype('int64').tolist()
    data=[x,y]
    if config.mask_x:
        data.append(mask_x)
    if config.mask_y:
        data.append(mask_y)
    data = tuple(data)
    return data

def get_minibatches_idx(datastream, shuffle=False):
    train_X, valid_X, test_X, train_Y, valid_Y, test_Y = datastream
    minibatch_size=config.batch_size
    valid_minibatch_size=config.valid_batch_size
    try:
        n_train = train_X.get_value().shape[0]
        n_valid = valid_X.get_value().shape[0]
        n_test= test_X.get_value().shape[0]
    except:
        n_train = len(train_X)
        n_valid = len(valid_X)
        n_test = len(test_X)

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
    train_minibatches=get_(n_train,minibatch_size)
    valid_minibatches = get_(n_valid, valid_minibatch_size)
    test_minibatches = get_(n_test, valid_minibatch_size)
    return train_minibatches,valid_minibatches,test_minibatches

def get_sample_data(datastream):
    train_X, valid_X, test_X, train_Y, valid_Y, test_Y = datastream
    try:
        n_train = train_X.get_value().shape[0]
    except:
        n_train = len(train_X)
    index = [config.rng.randint(0, n_train)]
    data_x=train_X
    data_y=train_Y
    if not config.transpose_x:
        x = [data_x[t] for t in index]
    else:
        x = [data_x[t] for t in index]
        x = np.array(x)
        mask_x = np.ones([len(index), len(x[0])]).astype(theano.config.floatX)
        x_new = []
        for idx in range(len(x[0])):
            x_new.append([x[i][idx] for i in range(len(x))])
        x = x_new
        mask_x=mask_x.transpose()
    if not config.transpose_y:
        y = [data_y[t] for t in index]
    else:
        y = [data_y[t] for t in index]
        y = np.array(y)

        mask_y = np.ones([len(index), len(y[0])]).astype(theano.config.floatX)
        y_new = []
        for idx in range(len(y[0])):
            y_new.append([y[i][idx] for i in range(len(y))])
        y = y_new
        mask_y=mask_y.transpose()
    if config.int_x:x=np.asarray(x).astype('int64').tolist()
    if config.int_y:y=np.asarray(y).astype('int64').tolist()
    data=[x,y]
    if config.mask_x:
        data.append(mask_x)
    if config.mask_y:
        data.append(mask_y)
    data = tuple(data)
    return data

def print_config(model, algrithm, extension):
    logger('Configurations:',0,1)
    logger('config:', 1)
    for key in config.__dict__ :
        if not key.startswith('__'):
            logger( key+' : %s'%config.__dict__[key],2)
    logger('model:', 1)
    for key in model.__dict__:
        if not key.startswith('__'):
            logger(key + ' : %s'%model.__dict__[key], 2)
    logger('layer:', 1)
    for lykey in model.layers:
        logger(lykey+":", 2)
        for key in model.layers[lykey].__dict__:
            if not key.startswith('__'):
                logger(key + ' : %s'% model.layers[lykey].__dict__[key], 3)
    logger('algrithm:', 1)
    logger(str(algrithm.config.__class__), 2)
    for key in algrithm.config.__dict__:
        if not key.startswith('__'):
            logger(key + ' : %s'% algrithm.config.__dict__[key], 3)
    logger('extension:', 1)
    for ex in extension:
        logger(str(ex.__class__),2)
        for key in ex.config.__dict__:
            if not key.startswith('__'):
                logger(key + ' : %s'% ex.config.__dict__[key], 3)

def get_modelstream(model,algrithm,get_fn=True):
    logger("Building Model:",0,1)
    NNB_model = model
    train_inputs=NNB_model.train_inputs
    NNB_model.get_cost_pred_error()
    params = NNB_model.params
    cost = NNB_model.cost
    error=NNB_model.error
    pred_Y=NNB_model.pred_Y
    optimizer = algrithm.config
    optimizer.init(params,cost)
    train_updates=optimizer.get_updates()
    model.updates=train_updates
    debug_output=[]
    for key in NNB_model.layers:
        debug_output.extend(NNB_model.layers[key].debug_stream)
    train_output=[cost]
    if get_fn:
        logger('Compiling Training Model',1)
        train_model = theano.function(inputs=train_inputs,
                                      outputs=train_output,
                                      updates=train_updates)
        logger('Compiling Validing Model',1)
        valid_model = theano.function(inputs=train_inputs,
                                      outputs=error)
        logger('Compiling Test Model',1)
        test_model = theano.function(inputs=train_inputs,
                                     outputs=error)
        logger('Compiling Sampling Model',1)
        sample_model = theano.function(inputs=train_inputs,
                                     outputs=[pred_Y,cost,error])
        logger('Compiling Model',1)
        model = theano.function(inputs=train_inputs,
                                outputs=pred_Y,on_unused_input='ignore')
        return [train_model, valid_model, test_model, sample_model,model, [],optimizer]
    else:
        return [None, None, None, None,None, [],optimizer]




