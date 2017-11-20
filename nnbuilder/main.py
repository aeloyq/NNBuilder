# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 19:37:12 2016

@author: aeloyq
"""


class Config:
    def __init__(self):
        self.name = 'unamed'
        self.batch_size = 20
        self.valid_batch_size = 64
        self.max_epoch = 50
        self.savelog = True
        self.verbose = 5

    def set(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

    def is_log_detail(self):
        return self.verbose < 0 or self.verbose >= 5

    def is_log_inline(self):
        return self.verbose > 1 and self.verbose < 5

    def is_log_silent(self):
        return self.verbose in [0, 1]


config = Config()
import numpy as np
import types
import copy
import os
import tools
import extensions
from logger import logger
from collections import OrderedDict
from nnbuilder.kernel import *
from extensions import monitor
from nnbuilder.optimizers import gradientdescent


class DataFrame:
    def __init__(self, train=None, valid=None, test=None, size=None, auto_split=0.2, stream=None):
        self.train = train
        self.valid = valid
        self.test = test
        self.size = size
        self.valid_size = 0
        self.test_size = 0
        self.auto_split = auto_split
        self.stream = stream
        self._buffer = None
        self._indices_valid = False
        self._indices_test = False
        self.prepare()

    def prepare(self):
        # determine whether to read stream
        if self.train is not None:
            self._read_stream = False
        elif self.stream is not None:
            self._read_stream = True
        else:
            assert AssertionError('Data not given!')
        # prepare data
        if not self._read_stream:
            self.size = len(self.train[1])
            if self.valid is None:
                valid_idx = self.auto_split_indices()
                self.valid = [self.train[0][valid_idx], self.train[1][valid_idx]]
                self.valid_size = len(valid_idx)
                test_idx = self.auto_split_indices()
                self.test = [self.train[0][test_idx], self.train[1][test_idx]]
                self.test_size = len(test_idx)
            else:
                self.valid_size = len(self.valid[1])
                self.test_size = len(self.test[1])
        else:
            if self.size is None:
                assert AssertionError('Size not given!')
            else:
                if self.valid is None:
                    self.valid = self.auto_split_indices()
                    self.valid_size = len(self.valid)
                    self._indices_valid = True
                else:
                    self.valid_size = len(self.valid[1])
                if self.test is None:
                    self.test = self.auto_split_indices()
                    self.test_size = len(self.test)
                    self._indices_test = True
                else:
                    self.test_size = len(self.test[1])

    @staticmethod
    def get_batch_indices(num, batchsize):
        n_minibatches = (num - 1) // batchsize + 1
        return range(n_minibatches)

    @staticmethod
    def get_minibatch_indices(batch_index, num, batchsize):
        return np.arange(batch_index * batchsize, min(num, (batch_index + 1) * batchsize))

    def get_train(self):
        if not self._read_stream:
            return self.train
        else:
            train, self._buffer = self.stream(range(self.size), self._buffer)
            return train

    def get_single_train(self, index):
        if not self._read_stream:
            return [self.train[0][[index]], self.train[1][[index]]]
        else:
            train_minibatch, _ = self.stream([index], self._buffer)
            return train_minibatch

    def get_minibatch_train(self, batch_index):
        minibatch_indices = self.get_minibatch_indices(batch_index, self.size, config.batch_size)
        if not self._read_stream:
            return [self.train[0][minibatch_indices], self.train[1][minibatch_indices]]
        else:
            train, self._buffer = self.stream(minibatch_indices, self._buffer)
            return train

    def get_valid(self):
        if not self._read_stream or not self._indices_valid:
            return self.valid
        else:
            valid, _ = self.stream(range(self.valid_size), self._buffer)
            return valid

    def get_single_valid(self, index):
        if not self._read_stream:
            return [self.valid[0][[index]], self.valid[1][[index]]]
        else:
            valid, _ = self.stream([index], self._buffer)
            return valid

    def get_minibatch_valid(self, batch_index):
        minibatch_indices = self.get_minibatch_indices(batch_index, self.valid_size, config.valid_batch_size)
        if not self._read_stream or not self._indices_valid:
            return [self.valid[0][minibatch_indices], self.valid[1][minibatch_indices]]
        else:
            valid_minibatch, _ = self.stream(minibatch_indices, self._buffer)
            return valid_minibatch

    def get_test(self):
        if not self._read_stream or not self._indices_test:
            return self.test
        else:
            test, _ = self.stream(range(self.test_size), self._buffer)
            return test

    def get_single_test(self, index):
        if not self._read_stream:
            return [self.test[0][[index]], self.test[1][[index]]]
        else:
            test, _ = self.stream([index], self._buffer)
            return test

    def get_minibatch_test(self, batch_index):
        minibatch_indices = self.get_minibatch_indices(batch_index, self.test_size, config.valid_batch_size)
        if not self._read_stream or not self._indices_test:
            return [self.test[0][minibatch_indices], self.test[1][minibatch_indices]]
        else:
            test_minibatch, _ = self.stream(minibatch_indices, self._buffer)
            return test_minibatch

    def auto_split_indices(self):
        return np.sort(np.random.permutation(self.size)[:np.round(self.size * self.auto_split)])


class MainLoop:
    def __init__(self):
        '''

        '''
        pass

    @staticmethod
    def train(data, model, optimizer=gradientdescent.sgd, extensions=(monitor,), verbose=None):
        '''

        :param data:
        :param model:
        :param optimizer:
        :param extensions:
        :param stream:
        :param stream_load:
        :return:
        '''
        if verbose is not None:
            config.verbose = verbose
        # Prepare train
        MainLoop.init_nnb()
        MainLoop.init_datas(data)
        model.build()
        model.optimize(optimizer)
        MainLoop.print_config(model, optimizer, extensions)
        model.compile()
        logger("Train   Model", 0, 1)
        max_epoch = config.max_epoch
        train_history = {}
        train_history['name'] = config.name
        train_history['n_epoch'] = 0
        train_history['n_iter'] = 0
        train_history['iter'] = 0
        train_history['stop'] = False
        train_history['train_loss'] = 1
        train_history['train_losses'] = []
        train_history['losses'] = []
        train_history['scores'] = OrderedDict()
        for m in model.metrics:
            train_history['scores'][m.name] = []
        for ex in extensions:
            ex.build(MainLoop, model, data, extensions, config, logger, train_history)

        # Main
        logger('Training Start', 1)
        for ex in extensions:   ex.before_train()
        if train_history['stop']:
            logger("Training Finished", 1, 1)
            return
        while (True):
            # Prepare data
            train_history['minibatch_list'] = DataFrame.get_batch_indices(data.size, config.batch_size)
            # Stop When Timeout
            if train_history['n_epoch'] > max_epoch - 1 and max_epoch != -1:
                logger("Training Finished", 1, 1)
                break
            # Iniate test
            if train_history['n_iter'] == 0:
                for ex in extensions:   ex.before_init_iter()
                MainLoop.test(model, data, train_history)
                for ex in extensions:   ex.after_init_iter()

            # Train model iter by iter

            for ex in extensions:   ex.before_epoch()

            for iter, index in enumerate(train_history['minibatch_list']):
                train_history['iter'] = iter

                for ex in extensions:   ex.before_iteration()

                train_X, train_Y = data.get_minibatch_train(index)
                train_history['train_loss'] = model.train(train_X, train_Y)
                train_history['n_iter'] += 1

                for ex in extensions:   ex.after_iteration()

                # After epoch
                if (iter == (data.size - 1) // config.batch_size):
                    train_history['train_losses'].append(train_history['train_loss'])
                    train_history['n_epoch'] += 1
                    MainLoop.test(model, data, train_history)

                    for ex in extensions:   ex.after_epoch()

                # Stop when needed
                if train_history['stop']:
                    for ex in extensions:   ex.after_train()
                    return

        for ex in extensions:   ex.after_train()

        logger("Finished", 0, 1)

        return train_history

    @staticmethod
    def valid(model, data):
        minibatch_list = DataFrame.get_batch_indices(data.valid_size, config.valid_batch_size)
        result = {'loss': []}
        for i, m in enumerate(model.metrics):
            result[m.name] = []
        for index in minibatch_list:
            valid_X, valid_Y = data.get_minibatch_valid(index)
            valid_result = model.valid(valid_X, valid_Y)
            result['loss'].append(valid_result['loss'])
            for i, m in enumerate(model.metrics):
                result[m.name].append(valid_result[m.name])
        result['loss'] = np.mean(result['loss'])
        for i, m in enumerate(model.metrics):
            result[m.name] = np.mean(result[m.name])
        return result

    @staticmethod
    def test(model, data, train_history, write_history=True):
        minibatch_list = DataFrame.get_batch_indices(data.test_size, config.valid_batch_size)
        result = {'loss': []}
        for i, m in enumerate(model.metrics):
            result[m.name] = []
        for index in minibatch_list:
            test_X, test_Y = data.get_minibatch_test(index)
            test_result = model.valid(test_X, test_Y)
            result['loss'].append(test_result['loss'])
            for i, m in enumerate(model.metrics):
                result[m.name].append(test_result[m.name])
        result['loss'] = np.mean(result['loss'])
        for i, m in enumerate(model.metrics):
            result[m.name] = np.mean(result[m.name])
        if write_history:
            train_history['losses'].append(result['loss'])
            for i, m in enumerate(model.metrics):
                train_history['scores'][m.name].append(result[m.name])
        return result

    @staticmethod
    def debug(data, model, optimizer=gradientdescent.sgd):
        model.build()
        model.optimize(optimizer)
        model.compile(optimizer)
        extensions.debug.build(MainLoop, model, data, [], config, logger, {})
        return extensions.debug.config.debug()

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
        if not os.path.exists('./%s/save/final' % config.name):
            os.mkdir('./%s/save/final' % config.name)
        if not os.path.exists('./%s/save/valid' % config.name):
            os.mkdir('./%s/save/valid' % config.name)
        if not os.path.exists('./%s/save/valid/best' % config.name):
            os.mkdir('./%s/save/valid/best' % config.name)
        if not os.path.exists('./%s/plot' % config.name):
            os.mkdir('./%s/plot' % config.name)
        if not os.path.exists('./%s/plot/model' % config.name):
            os.mkdir('./%s/plot/model' % config.name)
        if not os.path.exists('./%s/plot/progress' % config.name):
            os.mkdir('./%s/plot/progress' % config.name)

    @staticmethod
    def init_datas(data):
        logger("Process Data", 0, 1)
        v_num = data.valid_size
        t_num = data.test_size
        v_batches = (v_num - 1) // config.valid_batch_size + 1
        t_batches = (t_num - 1) // config.valid_batch_size + 1
        info = [['Datasets', '|', 'Train', '|', 'Valid', '|', 'Test']]
        strip = [15, 1, 22, 1, 22, 1, 22]
        info.append(
            ["=" * strip[0], "|" * strip[1], "=" * strip[2], "|" * strip[3], "=" * strip[4], "|" * strip[5],
             "=" * strip[6]])
        num = data.size
        batches = (num - 1) // config.batch_size + 1
        info.append(['Detail', '|', '{}*{}'.format(batches, config.batch_size), '|', '{}*{}'.format(v_batches,
                                                                                                    config.valid_batch_size),
                     '|', '{}*{}'.format(t_batches, config.valid_batch_size)])
        info.append(['Total', '|', '{}'.format(num), '|', '{}'.format(v_num), '|', '{}'.format(t_num)])
        if config.is_log_detail():
            logger(tools.printer.paragraphformatter(info, LengthList=strip, Align='center'), 1)

    @staticmethod
    def print_config(model, optimizer, extension):
        def get_info(key, item, column, truthvalue=True, stringvalue=True):
            if key.startswith('_'):
                pass
            elif type(item) == int or type(item) == float:
                column.append('{} = {}'.format(key, item))
            elif type(item) == types.BooleanType and truthvalue:
                column.append('{} = {}'.format(key, item))
            elif type(item) == types.StringType and stringvalue and item.strip() != "":
                column.append('{} = {}'.format(key, item))

        strip = [0, 0, 28, 1, 28, 1, 28]

        logger('Build   Model', 0, 1)
        info_all = []
        info_all.append(['', ' ', 'Global', '|', 'Graph', '|', 'Extension'])
        info_all.append(
            [" " * strip[0], " " * strip[1], "=" * strip[2], "|" * strip[3], "=" * strip[4], "|" * strip[5],
             "=" * strip[6]])

        first_column_info = []
        for key in config.__dict__:
            if not key.startswith('__'):
                get_info(key, config.__dict__[key], first_column_info)
        first_column_info.extend(["", "Model", "=" * strip[4]])
        tmp_column_info = []
        for key in model.__dict__:
            if not key.startswith('__'):
                get_info(key, model.__dict__[key], tmp_column_info)
        first_column_info.extend(tmp_column_info)

        first_column_info.extend(["", "Optimizer:%s" % (optimizer.__class__.__name__), "=" * strip[-1]])

        for key in optimizer.__dict__:
            if not key.startswith('__'):
                get_info(key, optimizer.__dict__[key], first_column_info)

        second_column_info = []

        for lykey in model.layers:
            second_column_info.extend([lykey + " " * strip[-1], "-" * strip[-1]])
            for key in model.layers[lykey].__dict__:
                if not key.startswith('__'):
                    get_info(key, model.layers[lykey].__dict__[key], second_column_info, truthvalue=False)
            second_column_info.append("")

        third_column_info = []
        for ex in extension:
            third_column_info.extend([ex.__class__.__name__ + " " * strip[-1], "-" * strip[-1]])
            for key in ex.__dict__:
                get_info(key, ex.__dict__[key], third_column_info)
            third_column_info.append("")
        if third_column_info != []:
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
        if config.is_log_detail():
            logger(tools.printer.paragraphformatter(info_all, LengthList=strip, Align='center'), 1)


# Shortcuts
train = MainLoop.train
debug = MainLoop.debug
