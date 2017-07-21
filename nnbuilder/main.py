# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:37:12 2016

@author: aeloyq
"""

import theano
import numpy as np
import config
import types
import copy
import os
import tools
import extensions
from logger import logger
from collections import OrderedDict


class mainloop:
    '''

    '''

    def __init__(self):
        '''

        '''
        pass

    @staticmethod
    def train(data, model, algrithm, extensions, stream=None, stream_stdin_func=np.load, prt_conf=True):
        '''

        :param data: 
        :param model:
        :param algrithm:
        :param extensions:
        :param stream:
        :param stream_stdin_func:
        :param prt_conf:
        :return:
        '''
        # Prepare train
        logger("Prepare Model:", 0, 1)
        mainloop.init_nnb()
        mainloop.build_model(model)
        n_data = mainloop.init_datas(data, stream, stream_stdin_func)
        if prt_conf: mainloop.print_config(model, algrithm, extensions)
        fn_dict = mainloop.get_functions(model, algrithm)
        logger("Trainning Model:", 0, 1)
        max_epoch = config.max_epoch
        kwargs = {}
        kwargs['model'] = model
        kwargs['datas'] = data
        kwargs['stream'] = stream
        kwargs['optimizer'] = algrithm.config
        kwargs['logger'] = logger
        kwargs['config'] = config
        kwargs['batchsize'] = config.batch_size
        kwargs['n_data'] = n_data
        kwargs['train_cost'] = 1
        kwargs['test_error'] = 1
        kwargs['debug_result'] = []
        kwargs['n_epoch'] = 0
        kwargs['n_iter'] = 0
        kwargs['n_part'] = 0
        kwargs['iter'] = 0
        kwargs['errors'] = []
        kwargs['costs'] = []
        kwargs['stop'] = False
        kwargs.update(fn_dict)
        extension_instance = []
        for ex in extensions: ex.config.kwargs = kwargs;ex.config.init();extension_instance.append(ex.config)
        kwargs['extensions'] = extensions

        # Main
        logger('Training Start', 1)
        for ex in extension_instance:   ex.before_train()
        if kwargs['stop']:
            return
        while (True):
            # Prepare data
            datas = mainloop.get_datas(data, stream, stream_stdin_func, kwargs['n_part'])
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
            kwargs['pre_iter'] = np.sum(kwargs['n_data'][1][:kwargs['n_part']])
            kwargs['prefix'] = kwargs['iter']
            for iter, index in enumerate(minibatches):
                kwargs['iter'] = iter + kwargs['prefix']

                for ex in extension_instance:   ex.before_iteration()

                d = mainloop.prepare_data(train_X, train_Y, index)
                traincost = kwargs['train_fn'](*d)
                kwargs['train_cost'] = traincost
                kwargs['n_iter'] += 1

                for ex in extension_instance:   ex.after_iteration()

                # After epoch
                if (kwargs['iter'] + 1 == kwargs['n_data'][1][kwargs['n_part']]):
                    if kwargs['stream'] == None or kwargs['n_part'] == len(kwargs['stream']) - 1:
                        kwargs['n_part'] = 0
                        kwargs['n_epoch'] += 1
                        testdatas = []
                        for index in kwargs['minibatches'][2]:
                            d = mainloop.prepare_data(test_X, test_Y, index)
                            testdatas.append(d)
                        test_result = np.array([kwargs['test_fn'](*tuple(testdata)) for testdata in testdatas])
                        kwargs['test_error'] = np.mean(test_result[:, 1])
                        kwargs['errors'].append(kwargs['test_error'])
                        kwargs['costs'].append(np.mean(test_result[:, 0]))

                        for ex in extension_instance:   ex.after_epoch()

                    else:
                        kwargs['n_part'] += 1

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
                    test_result = np.array([kwargs['test_fn'](*tuple(testdata)) for testdata in testdatas])
                    test_error = np.mean(test_result[:, 1])
                    if np.mean(test_error) == 0:
                        logger("●Trainning Sucess●", 1, 1)
                        break
            kwargs['iter'] = 0

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
    def debug(data,model,algrithm,use_data=None,get_function_only=False):
        if use_data is not None:
            tmp=[]
            for i in xrange(len(use_data)):
                if use_data[i]==1:
                    tmp.append(data[i])
            data=tmp
        mainloop.build_model(model)
        mainloop.build_optimizer(model,algrithm)
        if get_function_only:
            return data,mainloop.get_functions(model, algrithm)
        kwargs=OrderedDict()
        kwargs['logger']=logger
        kwargs['model']=model
        kwargs['data']=data
        extensions.debugmode.config.kwargs=kwargs
        extensions.debugmode.config.init()
        extensions.debugmode.config.debug()
        return kwargs, kwargs['debug_result']

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
        info = [['Datasets', '|', 'Train', '|', 'Valid', '|', 'Test']]
        strip = [15, 1, 22, 1, 22, 1, 22]
        if stream == None:
            info.append(["=" * strip[0], "|" * strip[1], "=" * strip[2], "|" * strip[3], "=" * strip[4], "|" * strip[5],
                         "=" * strip[6]])
            num = len(data[0])
            batches = (num - 1) // config.batch_size + 1
            info.append(['SingleFile', '|', '{}'.format(num), '|', '{}'.format(v_num), '|', '{}'.format(t_num)])
            info.append(['N_Batch', '|', '{}*{}'.format(batches, config.batch_size), '|', '{}*{}'.format(v_batches,
                                                                                                         config.valid_batch_size),
                         '|', '{}*{}'.format(t_batches,
                                             config.valid_batch_size)])
            logger(tools.printer.paragraphformatter(info, LengthList=strip, Align='center'), 1)
            return [[num], [batches]]
        else:
            v_num = len(data[0])
            t_num = len(data[1])
            v_batches = (v_num - 1) // config.valid_batch_size + 1
            t_batches = (t_num - 1) // config.valid_batch_size + 1
            n_data = [[], []]
            i = 0
            logger(tools.printer.lineformatter(info[0], LengthList=strip, Align='center'), 1)
            logger(tools.printer.lineformatter(
                ["=" * strip[0], "|" * strip[1], "=" * strip[2], "|" * strip[3], "=" * strip[4], "|" * strip[5],
                 "=" * strip[6]], LengthList=strip, Align='center'), 1)
            for part in stream:
                i += 1
                try:
                    d = stream_stdin_func(part)
                    num = len(d[0])
                    batches = (num - 1) // config.batch_size + 1
                    n_data[0].append(num)
                    n_data[1].append(batches)
                    logger(tools.printer.lineformatter(['Part {}'.format(i), '|', '{}/{} = {}'.format(num,
                                                                                                    config.batch_size,
                                                                                                    batches, ), '|',
                                                        '{}/{} = {}'.format(v_num,
                                                                          config.valid_batch_size,
                                                                          v_batches),
                                                        '|', '{}/{} = {}'.format(t_num,
                                                                               config.valid_batch_size,
                                                                               t_batches)], LengthList=strip,
                                                       Align='center'), 1)
                except:
                    logger("Broken part found in data stream !", 0)
            logger(tools.printer.lineformatter(
                ["-" * strip[0], "|" * strip[1], "-" * strip[2], "|" * strip[3], "-" * strip[4], "|" * strip[5],
                 "-" * strip[6]], LengthList=strip, Align='center'), 1)
            logger(
                tools.printer.lineformatter(['Total Data'.format(i), '|', '{}/{} = {}'.format(int(str(np.sum(n_data[0]))),
                                                                                            config.batch_size,
                                                                                            int(str(
                                                                                                np.sum(n_data[1]))), ),
                                             '|',
                                             '{}/{} = {}'.format(v_num,
                                                               config.valid_batch_size,
                                                               v_batches),
                                             '|', '{}/{} = {}'.format(t_num,
                                                                    config.valid_batch_size,
                                                                    t_batches)], LengthList=strip,
                                            Align='center'), 1)

            return n_data

    @staticmethod
    def print_config(model, algrithm, extension):
        def get_info(key, item, column, truthvalue=True, stringvalue=True):
            if type(item) == int or type(item) == float:
                column.append('{} = {}'.format(key, item))
            elif type(item) == types.BooleanType and truthvalue:
                column.append('{} = {}'.format(key, item))
            elif type(item) == types.StringType and stringvalue and item.strip() !="":
                column.append('{} = {}'.format(key, item))

        strip = [11, 1, 28, 1, 28, 1, 28]

        logger('Config Detail:', 0, 1)
        info_all = []
        info_all.append(['', ' ', 'Global', '|', 'Graph', '|', 'Extension'])
        info_all.append([" " * strip[0], " " * strip[1], "=" * strip[2], "|" * strip[3], "=" * strip[4], "|" * strip[5],
                         "=" * strip[6]])

        first_column_info = []
        for key in config.__dict__:
            if not key.startswith('__'):
                get_info(key, config.__dict__[key], first_column_info)
        first_column_info.extend(["","", "Model", "=" * strip[4]])
        tmp_column_info = []
        for key in model.__dict__:
            if not key.startswith('__'):
                get_info(key, model.__dict__[key], tmp_column_info)
        first_column_info.extend(tmp_column_info)

        second_column_info = []

        for lykey in model.layers:
            second_column_info.extend([lykey + " " * strip[-1], "-" * strip[-1]])
            for key in model.layers[lykey].__dict__:
                if not key.startswith('__'):
                    get_info(key, model.layers[lykey].__dict__[key], second_column_info, truthvalue=False)
            second_column_info.append("")

        second_column_info.extend(["","Algrithm","="* strip[-1]])

        for key in algrithm.config.__dict__:
            if not key.startswith('__'):
                get_info(key, algrithm.config.__dict__[key], second_column_info)

        third_column_info = []
        for ex in extension:
            third_column_info.extend([ex.__name__.split('.')[-1] + " " * strip[-1], "-" * strip[-1]])
            for key in ex.config.__dict__:
                get_info(key, ex.config.__dict__[key], third_column_info)
            third_column_info.append("")
        third_column_info.pop(-1)



        for i in range(max(len(first_column_info), len(second_column_info), len(third_column_info))):
            info_all.append([''])
            info_all[i + 2].append(' ')
            if i < len(first_column_info):
                info_all[i + 2].append(first_column_info[i])
            else:
                info_all[i + 2].append("")
            info_all[i + 2].append('|')
            if i < len(second_column_info):
                info_all[i + 2].append(second_column_info[i])
            else:
                info_all[i + 2].append("")
            info_all[i + 2].append('|')
            if i < len(third_column_info):
                info_all[i + 2].append(third_column_info[i])
            else:
                info_all[i + 2].append("")

        logger(tools.printer.paragraphformatter(info_all, LengthList=strip, Align='center'), 0)

    @staticmethod
    def get_datas(data, stream=None, stream_stdin_func=None, n_part=0):
        if stream == None:
            return data
        else:
            trainning_data = stream_stdin_func(stream[n_part])
            data_ = [trainning_data[0], data[0], data[1], trainning_data[1], data[2], data[3]]
            return data_

    @staticmethod
    def build_model(model):
        model.build()

    @staticmethod
    def build_optimizer(model,algrithm):
        params = model.params
        cost = model.cost
        optimizer = algrithm.config
        optimizer.init(params, cost)
        optimizer_updates = optimizer.get_updates()
        model_updates = model.updates
        model.optimizer_updates = optimizer_updates
        updates = model_updates.items() + optimizer_updates.items()
        raw_updates = model.raw_updates
        return updates,raw_updates,model_updates,optimizer_updates

    @staticmethod
    def get_functions(model, algrithm):
        logger("Building Function:", 0, 1)
        inputs = model.inputs
        cost = model.cost
        raw_cost = model.raw_cost
        error = model.error
        predict = model.predict
        updates, raw_updates, model_updates, optimizer_updates=mainloop.build_optimizer(model,algrithm)
        updates = model_updates.items() + optimizer_updates.items()
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
        return {'train_fn':train_fn,'valid_fn': valid_fn,'test_fn': test_fn,'sample_fn': sample_fn,'model_fn': model_fn}

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
        mask_x = None;
        mask_y = None
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
