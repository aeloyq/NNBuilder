# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import timeit
import numpy as np
import nnbuilder
import os
import theano.d3viz as d3v
import matplotlib
from extension import extension
from nnbuilder.layers.roles import *
from collections import OrderedDict
from nnbuilder.main import mainloop


class ex(extension):
    def __init__(self, kwargs):
        extension.__init__(self, kwargs)
        self.report_iter = False
        self.report_iter_frequence = 5
        self.report_epoch = True
        self.plot = False
        self.dotprint = True
        self.size = False

    def init(self):
        extension.init(self)
        self.batches = int(str(np.sum(self.kwargs['n_data'][1])))
        if self.plot:
            matplotlib.use('Agg')
            import matplotlib.pylab as plt
            global plt
            plt.cla()

    def before_train(self):
        kwargs = self.kwargs
        self.epoch_s_time = timeit.default_timer()
        self.iteration_s_time = timeit.default_timer()
        self.start_time = timeit.default_timer()
        self.path = './{}/tmp/'.format(nnbuilder.config.name)
        if not os.path.exists('./{}'.format(nnbuilder.config.name)): os.mkdir('./{}'.format(nnbuilder.config.name))
        if not os.path.exists('./{}/tmp'.format(nnbuilder.config.name)): os.mkdir(
            './{}/tmp'.format(nnbuilder.config.name))
        if self.dotprint: d3v.d3viz(kwargs['model'].output.output, self.path + 'model.html')

    def before_iteration(self):
        self.iter_start_time = timeit.default_timer()

    def after_iteration(self):
        if self.report_iter:
            n_iter = self.kwargs['n_iter']
            iter = self.kwargs['iter']+self.kwargs['pre_iter']
            process = ((iter + 1) * 100) / self.batches
            if n_iter % self.report_iter_frequence == 0:
                self.logger("Epoch:%d   Iter:%d   Bucket:%d   Time:%.2fs   " \
                            "Cost:%.4f      /%d%%/ Total:%ds" % (self.kwargs['n_epoch'], n_iter,self.kwargs['n_bucket'],
                                                                 timeit.default_timer() - self.iter_start_time,
                                                                 self.kwargs['train_cost'],
                                                                 process,
                                                                 (timeit.default_timer() - self.kwargs['start_time']) +
                                                                 self.kwargs['time']), 1)

    def plot_func(self, costs, errors, params, roles):
        x_axis = np.arange(len(costs)) + 1

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

        plt.cla()

    def after_epoch(self):
        kwargs = self.kwargs
        epoch_time = timeit.default_timer() - self.epoch_s_time
        self.epoch_s_time = timeit.default_timer()
        if self.report_epoch:
            self.logger("", 2)
            self.logger("◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆", 1)
            self.logger("  Single Epoch Done:", 1)
            self.logger("  Epoches:%d  " % kwargs['n_epoch'], 1)
            self.logger("  Iterations:%d" % (kwargs['n_iter']), 1)
            self.logger("  Time Used:%.2fs" % epoch_time, 1)
            self.logger("  Cost:%.4f   " % kwargs['costs'][-1], 1)
            self.logger("  Error:%.4f%%" % (kwargs['test_error'] * 100), 1)
            self.logger("◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆", 1)
            self.logger("", 2)
        if self.plot:
            p = OrderedDict()
            for key, param in kwargs['model'].params.items():
                p[key] = param.get_value()
            self.plot_func(self.kwargs['costs'], self.kwargs['errors'], p, kwargs['model'].roles)

    def after_train(self):
        kwargs = self.kwargs
        test_minibatches = kwargs['minibatches'][2]
        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['datas']
        test_model = kwargs['test_fn']
        testdatas = []
        for index in test_minibatches:
            data = mainloop.prepare_data(test_X, test_Y, index)
            testdatas.append(data)
        test_result = np.array([test_model(*tuple(testdata)) for testdata in testdatas])
        test_error = float(str(np.mean(test_result[:, 1]) * 100))
        test_cost = float(str(np.mean(test_result[:, 0])))
        self.logger("", 0)
        self.logger("All Finished:", 0)
        self.logger("Trainning finished after epoch:%s" % kwargs['n_epoch'], 1)
        self.logger("Trainning finished at iteration:%s" % kwargs['n_iter'], 1)
        if kwargs['best_iter'] != -1:
            self.logger("Best iteration:%s" % kwargs['best_iter'], 1)
        self.logger("Time used in total:%.2fs" % kwargs['time'], 1)
        self.logger("Finall cost:%.4f" % test_cost, 1)
        if kwargs['best_valid_error'] != 1.:
            self.logger("Best valid error:%.4f%%" % (kwargs['best_valid_error'] * 100), 1)
        self.logger("Finall test error:%.4f%%" % (test_error), 1)


config = ex({})
