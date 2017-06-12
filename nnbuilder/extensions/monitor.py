# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import extension
import timeit
import numpy as np
import nnbuilder
import matplotlib
import os
import threading
import theano.d3viz as d3v
from nnbuilder.layers.roles import *

matplotlib.use('Agg')
import matplotlib.pylab as plt
from collections import OrderedDict

base = extension.extension

class ex(base):
    def __init__(self, kwargs):
        base.__init__(self, kwargs)
        self.report_iter = False
        self.report_iter_frequence = 5
        self.report_epoch = True
        self.plot = False
        self.dotprint = True

    def init(self):
        base.init(self)
        data = self.kwargs['data_stream'][0]
        try:
            n_train = data.get_value().shape[0]
        except:
            n_train = len(data)
        self.patience = n_train
        self.batches = (n_train - 1) // self.kwargs['conf'].batch_size + 1

    def before_train(self):
        kwargs = self.kwargs
        self.epoch_s_time = timeit.default_timer()
        self.iteration_s_time = timeit.default_timer()
        self.start_time = timeit.default_timer()
        self.path = './{}/tmp/'.format(nnbuilder.config.name)
        if not os.path.exists('./{}'.format(nnbuilder.config.name)): os.mkdir('./{}'.format(nnbuilder.config.name))
        if not os.path.exists('./{}/tmp'.format(nnbuilder.config.name)): os.mkdir(
            './{}/tmp'.format(nnbuilder.config.name))
        if self.dotprint: d3v.d3viz(kwargs['dim_model'].output.output, self.path + 'model.html')

    def after_iteration(self):
        pass
        iteration_time = timeit.default_timer() - self.iteration_s_time
        self.iteration_s_time = timeit.default_timer()
        if self.report_iter:

            iter = self.kwargs['iteration_total']
            idx=self.kwargs['idx']
            process=((idx+1)*100)/self.batches
            if iter % self.report_iter_frequence == 0:
                self.logger("Epoch:%d   Iter:%d   Time:%.2fs   " \
                            "Cost:%.4f      ▉ %d%% ▉ Total:%.2fs" % (self.kwargs['epoches'], iter,
                                           iteration_time, self.kwargs['train_result'],process,self.kwargs['time_used']), 2)



    def plot_func(self, costs, errors,params,roles):
        x_axis = np.arange(len(costs))+1

        plt.figure(1)
        plt.cla()
        plt.title(nnbuilder.config.name)
        plt.ylabel('Loss')
        plt.xlabel('Epochs')
        plt.plot(x_axis, costs, label='Loss', color='orange')
        plt.legend()
        plt.savefig(self.path + 'process_cost.png')

        plt.figure(2)
        plt.cla()
        plt.title(nnbuilder.config.name)
        plt.ylabel('Error')
        plt.xlabel('Epochs')
        plt.plot(x_axis, errors, label='Error', color='blue')
        plt.legend()
        plt.savefig(self.path + 'process_error.png')

        n_im = len(params)
        a = np.int(np.sqrt(n_im))
        b = a
        if a * b < n_im: a += 1
        if a * b < n_im: b += 1
        plt.figure(3, (b * 4, a * 4))
        plt.cla()

        i = 0
        for key, param in params.items():
            i += 1
            if roles[key] is weight:
                plt.subplot(a, b, i)
                value = param
                plt.title(key + ' ' + str(value.shape))
                img = np.asarray(value)
                if img.ndim != 1:
                    plt.imshow(img, cmap='gray')
            elif roles[key] is bias:
                plt.subplot(a, b, i)
                y = param
                plt.title(key + ' ' + str(y.shape))
                x_axis_bi = np.arange(y.shape[0])
                plt.plot(x_axis_bi, y, color='black')
        plt.savefig(self.path + 'paramsplot.png')

    def after_epoch(self):
        kwargs = self.kwargs
        epoch_time = timeit.default_timer() - self.epoch_s_time
        self.epoch_s_time = timeit.default_timer()
        if self.report_epoch:
            self.logger("", 2)
            self.logger("◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆", 2)
            self.logger("  Single Epoch Done:", 2)
            self.logger("  Epoches:%d  " % kwargs['epoches'], 2)
            self.logger("  Iterations:%d" % (kwargs['iteration_total']), 2)
            self.logger("  Time Used:%.2fs" % epoch_time, 2)
            self.logger("  Cost:%.4f   " % kwargs['costs'][-1], 2)
            self.logger("  Error:%.4f%%" % (kwargs['train_error'] * 100), 2)
            self.logger("◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆", 2)
            self.logger("", 2)
        if self.plot:
            costs = self.kwargs['costs']
            errors = self.kwargs['errors']
            p=OrderedDict()
            for key,param in kwargs['dim_model'].params.items():
                p[key]=param.get_value()
            self.plot_func(costs, errors,p,kwargs['dim_model'].roles)

    def after_train(self):
        kwargs = self.kwargs
        test_minibatches = kwargs['minibatches'][2]
        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['data_stream']
        test_model = kwargs['test_model']
        testdatas = []
        for _, index in test_minibatches:
            data = kwargs['prepare_data'](test_X, test_Y, index)
            testdatas.append(data)
        test_result = np.array([test_model(*tuple(testdata)) for testdata in testdatas])
        test_error = np.mean(test_result[:, 1])
        test_cost = np.mean(test_result[:, 0])
        self.logger("", 0)
        self.logger("All Finished:", 0)
        self.logger("Trainning finished after epoch:%s" % kwargs['epoches'], 1)
        self.logger("Trainning finished at iteration:%s" % kwargs['iteration_total'], 1)
        if kwargs['best_iter'] != -1:
            self.logger("Best iteration:%s" % kwargs['best_iter'], 1)
        self.logger("Time used in total:%.2fs" % kwargs['time_used'], 1)
        self.logger("Finall cost:%.4f" % test_cost, 1)
        if kwargs['best_valid_error'] != 1.:
            self.logger("Best valid error:%.4f%%" % (kwargs['best_valid_error'] * 100), 1)
        self.logger("Finall test error:%.4f%%" % (test_error * 100), 1)


config = ex({})
