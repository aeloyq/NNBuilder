# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:37:12 2016

@author: aeloyq
"""

import theano
import numpy as np
import config
import types
import timeit
import copy
import os
from logger import logger


class mainloop:
    def __init__(self):
        pass

    @staticmethod
    def train(data, model, algrithm, extensions, stream=None, stream_stdin_func=np.load, prt_conf=False):
        # Prepare train
        mainloop.init_nnb()
        n_data = mainloop.init_datas(data, stream, stream_stdin_func)
        model.build()
        if prt_conf: mainloop.print_config(model, algrithm, extensions)
        train_fn, valid_fn, test_fn, sample_fn, model_fn = mainloop.get_modelstream(model, algrithm)
        logger("Trainning Model:", 0, 1)
        max_epoch = config.max_epoch
        kwargs = {}
        kwargs['model'] = model
        kwargs['datas'] = data
        kwargs['stream'] = stream
        kwargs['optimizer'] = algrithm.config
        kwargs['logger'] = logger
        kwargs['train_fn'] = train_fn
        kwargs['valid_fn'] = valid_fn
        kwargs['test_fn'] = test_fn
        kwargs['sample_fn'] = sample_fn
        kwargs['model_fn'] = model_fn
        kwargs['config'] = config
        kwargs['batchsize'] = config.batch_size
        kwargs['n_data'] = n_data
        kwargs['train_cost'] = 1
        kwargs['test_error'] = 1
        kwargs['debug_result'] = []
        kwargs['n_epoch'] = 0
        kwargs['n_iter'] = 0
        kwargs['n_bucket'] = 0
        kwargs['iter'] = 0
        kwargs['time'] = 0
        kwargs['errors'] = []
        kwargs['costs'] = []
        kwargs['stop'] = False
        kwargs['best_valid_error'] = 1
        kwargs['best_iter'] = -1
        extension_instance = []
        for ex in extensions: ex.config.kwargs = kwargs;ex.config.init();extension_instance.append(ex.config)
        kwargs['extensions'] = extension_instance

        # Main
        logger('Training Start', 1)
        for ex in extension_instance:   ex.before_train()
        if kwargs['stop']:
            return
        kwargs['start_time'] = timeit.default_timer()
        while (True):
            # Prepare data
            datas = mainloop.get_datas(data, stream, stream_stdin_func, kwargs['n_bucket'])
            kwargs['datas'] = datas
            train_X, valid_X, test_X, train_Y, valid_Y, test_Y = datas
            kwargs['minibatches'] = mainloop.get_minibatches(datas)
            # Stop When Timeout
            if kwargs['n_epoch'] > max_epoch - 1 and max_epoch != -1:
                logger("⊙Trainning Time Out⊙", 1, 1)
                break
            # Train model iter by iter

            for ex in extension_instance:   ex.before_epoch()

            minibatches = kwargs['minibatches'][0][kwargs['iter']:]
            kwargs['pre_iter']=np.sum(kwargs['n_data'][1][:kwargs['n_bucket']])
            kwargs['prefix']=kwargs['iter']
            for iter, index in enumerate(minibatches):
                kwargs['iter'] = iter+kwargs['prefix']

                for ex in extension_instance:   ex.before_iteration()

                d = mainloop.prepare_data(train_X, train_Y, index)
                traincost = train_fn(*d)
                kwargs['train_cost'] = traincost
                kwargs['n_iter'] += 1

                for ex in extension_instance:   ex.after_iteration()

                # After epoch
                if (kwargs['iter'] + 1 == kwargs['n_data'][1][kwargs['n_bucket']]):
                    if kwargs['stream'] == None or kwargs['n_bucket'] == len(kwargs['stream']) - 1:
                        kwargs['n_bucket'] = 0
                        kwargs['n_epoch'] += 1
                        testdatas = []
                        for index in kwargs['minibatches'][2]:
                            d = mainloop.prepare_data(test_X, test_Y, index)
                            testdatas.append(d)
                        test_result = np.array([test_fn(*tuple(testdata)) for testdata in testdatas])
                        kwargs['test_error'] = np.mean(test_result[:, 1])
                        kwargs['errors'].append(kwargs['test_error'])
                        kwargs['costs'].append(np.mean(test_result[:, 0]))

                        for ex in extension_instance:   ex.after_epoch()

                    else:
                        kwargs['n_bucket'] += 1

                # Stop when needed
                if kwargs['stop']:
                    for ex in extension_instance:   ex.after_train()
                    return

                # Stop When Sucess
                if traincost == 0:
                    testdatas = []
                    for index in kwargs['minibatches'][2]:
                        d = mainloop.prepare_data(test_X, test_Y, index)
                        testdatas.append(d)
                    test_result = np.array([test_fn(*tuple(testdata)) for testdata in testdatas])
                    test_error = np.mean(test_result[:, 1])
                    if np.mean(test_error) == 0:
                        kwargs['best_iter'] = kwargs['n_iter']
                        logger("●Trainning Sucess●", 1, 1)
                        break
            kwargs['iter'] = 0
        kwargs['time'] = kwargs['time'] + (timeit.default_timer() - kwargs['start_time'])

        for ex in extension_instance:   ex.after_train()

        return

    @staticmethod
    def use(model):
        model.build()
        NNB_model = model
        inputs = NNB_model.inputs
        params = NNB_model.params
        cost = NNB_model.cost
        raw_cost = NNB_model.raw_cost
        error = NNB_model.error
        predict = NNB_model.predict
        return theano.function(inputs, predict)

    @staticmethod
    def debug(model):
        model.build()
        NNB_model = model
        inputs = NNB_model.inputs
        params = NNB_model.params
        cost = NNB_model.cost
        raw_cost = NNB_model.raw_cost
        error = NNB_model.error
        predict = NNB_model.predict
        return theano.function(inputs, predict)

    @staticmethod
    def init_nnb():
        # generate documents
        if not os.path.exists('./%s' % config.name):
            os.mkdir('./%s' % config.name)
        if not os.path.exists('./%s/log' % config.name):
            os.mkdir('./%s/log' % config.name)
        if not os.path.exists('./%s/save' % config.name):
            os.mkdir('./%s/save' % config.name)
        if not os.path.exists('./%s/save/epoch' % config.name):
            os.mkdir('./%s/save/epoch' % config.name)
        if not os.path.exists('./%s/tmp' % config.name):
            os.mkdir('./%s/tmp' % config.name)

    @staticmethod
    def init_datas(data, stream, stream_stdin_func):
        logger("Data Detail:", 0, 1)
        v_num = len(data[1])
        t_num = len(data[2])
        v_batches = (v_num - 1) // config.valid_batch_size + 1
        t_batches = (t_num - 1) // config.valid_batch_size + 1
        logger("Datasets        -          Train          -          Valid          -          test", 1)
        if stream == None:
            num = len(data[0])
            batches = (num - 1) // config.batch_size + 1
            logger("SingleFile        -          {}          -          {}          -          {}".format(num, v_num,
                                                                                                          t_num), 1)
            logger(
                "N_Batch           -         {}*{}         -         {}*{}         -         {}*{}".format(batches,
                                                                                                           config.batch_size,
                                                                                                           v_batches,
                                                                                                           config.valid_batch_size,
                                                                                                           t_batches,
                                                                                                           config.valid_batch_size),
                1)
            return [[num], [batches]]
        else:
            n_data = [[], []]
            i=0
            for bucket in stream:
                i+=1
                try:
                    d = stream_stdin_func(bucket)
                    num = len(d[0])
                    batches = (num - 1) // config.batch_size + 1
                    n_data[0].append(num)
                    n_data[1].append(batches)
                    logger("Bucket {}        -      {}/{}={}      -      {}/{}={}      -      {}/{}={}".format(i,num,config.batch_size,batches,
                                                                                                                  v_num,config.valid_batch_size,v_batches,
                                                                                                                  t_num,config.valid_batch_size,t_batches),
                           1)
                except:
                    logger("Broken bucket found in data stream !", 0)
            return n_data

    @staticmethod
    def print_config(model, algrithm, extension):
        def get_info(item):
            if type(item) == types.ObjectType:
                return 'object'
            elif type(item) == types.ClassType:
                return 'class'
            elif type(item) == types.InstanceType:
                return 'instanse'
            elif type(item) == types.FunctionType:
                return 'function'
            elif type(item) == types.ModuleType:
                return 'Module'
            else:
                return item

        logger('Configurations:', 0, 1)
        logger('config:', 1)
        for key in config.__dict__:
            if not key.startswith('__'):
                info = get_info(config.__dict__[key])
                logger(key + ' : %s' % info, 2)
        logger('model:', 1)
        for key in model.__dict__:
            if not key.startswith('__'):
                info = get_info(model.__dict__[key])
                logger(key + ' : %s' % info, 2)
        logger('layer:', 1)
        for lykey in model.layers:
            logger(lykey + ":", 2)
            for key in model.layers[lykey].__dict__:
                if not key.startswith('__'):
                    info = get_info(model.layers[lykey].__dict__[key])
                    logger(key + ' : %s' % info, 3)
        logger('algrithm:', 1)
        logger(str(algrithm.__name__.split('.')[-1]), 2)
        for key in algrithm.config.__dict__:
            if not key.startswith('__'):
                logger(key + ' : %s' % algrithm.config.__dict__[key], 3)
        logger('extension:', 1)
        for ex in extension:
            logger(str(ex.__name__.split('.')[-1]), 2)
            for key in ex.config.__dict__:
                if not key.startswith('__'):
                    logger(key + ' : %s' % ex.config.__dict__[key], 3)

    @staticmethod
    def get_datas(data, stream=None, stream_stdin_func=None, n_bucket=0):
        if stream == None:
            return data
        else:
            trainning_data = stream_stdin_func(stream[n_bucket])
            data_ = [trainning_data[0], data[0], data[1], trainning_data[1], data[2], data[3]]
            return data_

    @staticmethod
    def get_modelstream(model, algrithm):
        logger("Building Model:", 0, 1)
        NNB_model = model
        inputs = NNB_model.inputs
        params = NNB_model.params
        cost = NNB_model.cost
        raw_cost = NNB_model.raw_cost
        error = NNB_model.error
        predict = NNB_model.predict
        optimizer = algrithm.config
        optimizer.init(params, cost)
        train_updates = optimizer.get_updates()
        model_updates = NNB_model.updates
        NNB_model.train_updates = train_updates
        debug_output = []
        for key in NNB_model.layers:
            debug_output.extend(NNB_model.layers[key].debug_stream)
        updates = model_updates.items() + train_updates.items()
        raw_updates = model.raw_updates
        logger('Compiling Training Model', 1)
        train_fn = theano.function(inputs=inputs,
                                   outputs=cost,
                                   updates=updates)
        logger('Compiling Validing Model', 1)
        valid_fn = theano.function(inputs=inputs,
                                   outputs=error,
                                   updates=raw_updates)
        logger('Compiling Test Model', 1)
        test_fn = theano.function(inputs=inputs,
                                  outputs=[raw_cost, error],
                                  updates=raw_updates)
        logger('Compiling Sampling Model', 1)
        sample_fn = theano.function(inputs=inputs,
                                    outputs=[predict, raw_cost, error],
                                    updates=raw_updates)
        logger('Compiling Model', 1)
        model_fn = theano.function(inputs=inputs,
                                   outputs=predict, on_unused_input='ignore',
                                   updates=raw_updates)
        return [train_fn, valid_fn, test_fn, sample_fn, model_fn]

    @staticmethod
    def get_minibatches(data, shuffle=False, window=None):
        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = data
        batch_size = config.batch_size
        valid_minibatch_size = config.valid_batch_size
        try:
            n_train = len(train_X)
            n_valid = len(valid_X)
            n_test = len(test_X)
        except:
            n_train = train_X.get_value().shape[0]
            n_valid = valid_X.get_value().shape[0]
            n_test = test_X.get_value().shape[0]

        def arrange(num, batch_size, shuffle=False, window=None):
            # Prepare index list
            if shuffle:
                index_list = []
                if not window: window = batch_size * 100
                n_block = (num - 1) // window + 1
                idx_l = np.arange(n_block, dtype="int32")
                np.random.shuffle(idx_l)
                for i in idx_l:
                    nd = window
                    if i == n_block - 1: nd = num - (n_block - 1) * window
                    idxs = np.arange(nd, dtype="int32") + i * window
                    np.random.shuffle(idxs)
                    index_list.extend(idxs)
            else:
                index_list = np.arange(num, dtype="int32")

            # Arrange batches
            minibatches = []
            minibatches_shuffle = range((num - 1) // batch_size)
            if shuffle: np.random.shuffle(minibatches_shuffle)
            for i in range((num - 1) // batch_size):
                idx = minibatches_shuffle[i] * batch_size
                minibatches.append(index_list[idx:
                idx + batch_size])

            # Make a minibatch out of what is left
            minibatches.insert(np.random.randint((num - 1) // batch_size),
                               index_list[((num - 1) // batch_size) * batch_size:])

            return minibatches

        train_minibatches = arrange(n_train, batch_size, shuffle, window)
        valid_minibatches = arrange(n_valid, valid_minibatch_size)
        test_minibatches = arrange(n_test, valid_minibatch_size)
        return train_minibatches, valid_minibatches, test_minibatches

    @staticmethod
    def prepare_data(data_x, data_y, index):
        mask_x=None;mask_y=None
        x = copy.deepcopy([data_x[t] for t in index])
        y = copy.deepcopy([data_y[t] for t in index])
        if config.transpose_x:
            maxlen = max([len(d) for d in x])
            x = np.array(x)
            mask_x = np.ones([len(index), maxlen]).astype('int16')
            for idx, i in enumerate(x):
                for j in range(len(i), maxlen):
                    i.append(np.zeros_like(i[0]).tolist())
                    mask_x[idx, j] = 0
            x_new = []
            for idx in range(len(x[0])):
                x_new.append([x[i][idx] for i in range(len(x))])
            x = x_new
            mask_x = mask_x.transpose()

        if config.transpose_y:
            maxlen = max([len(d) for d in y])
            y = np.array(y)
            mask_y = np.ones([len(index), maxlen]).astype('int16')
            for idx, i in enumerate(y):
                for j in range(len(i), maxlen):
                    i.append(np.zeros_like(i[0]).tolist())
                    mask_y[idx, j] = 0
            y_new = []
            for idx in range(len(y[0])):
                y_new.append([y[i][idx] for i in range(len(y))])
            y = y_new
            mask_y = mask_y.transpose()
        if config.int_x: x = np.asarray(x).astype('int64').tolist()
        if config.int_y: y = np.asarray(y).astype('int64').tolist()
        data = [x, y]
        if config.mask_x:
            data.append(mask_x)
        if config.mask_y:
            data.append(mask_y)
        data = tuple(data)
        return data




# Shortcuts
train = mainloop.train
use = mainloop.use
debug = mainloop.debug
