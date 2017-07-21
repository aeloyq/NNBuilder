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
from extension import extension
from nnbuilder.layers.roles import *
from collections import OrderedDict
from nnbuilder.main import mainloop
from nnbuilder.tools import printer


class ex(extension):
    def __init__(self, kwargs):
        extension.__init__(self, kwargs)
        self.report_iter = False
        self.report_iter_frequence = 5
        self.report_epoch = True
        self.plot = False
        self.dotprint = True

    def init(self):
        extension.init(self)
        self.batches = int(str(np.sum(self.kwargs['n_data'][1])))
        self.pre_time=[0,0]
        if self.plot:
            import nnbuilder.tools.plotter as plt
            global plt

    def before_train(self):
        k = self.kwargs
        self.start_time = timeit.default_timer()
        self.epoch_start_time = timeit.default_timer()
        self.iter_start_time = timeit.default_timer()
        self.report_start_time = timeit.default_timer()

        self.path = './{}/tmp/'.format(nnbuilder.config.name)
        if not os.path.exists('./{}'.format(nnbuilder.config.name)): os.mkdir('./{}'.format(nnbuilder.config.name))
        if not os.path.exists('./{}/tmp'.format(nnbuilder.config.name)): os.mkdir(
            './{}/tmp'.format(nnbuilder.config.name))
        if self.dotprint: d3v.d3viz(k['model'].output.output, self.path + 'model.html')

    def before_iteration(self):
        self.iter_start_time = timeit.default_timer()
        if self.kwargs['n_iter'] % self.report_iter_frequence == 0:
            self.report_start_time = timeit.default_timer()

    def after_iteration(self):
        if self.report_iter:
            n_iter = self.kwargs['n_iter']
            iter = self.kwargs['iter'] + self.kwargs['pre_iter']
            process = ((iter + 1) * 100) / self.batches
            if n_iter % self.report_iter_frequence == 0:
                k = self.kwargs
                report = [[], []]
                report[0].append("Epoch:{}".format(k['n_epoch'] + 1))
                report[1].append(10)
                report[0].append('|')
                report[1].append(1)

                if k['stream'] is not None:
                    report[0].append("Part:{}".format(k['n_part'] + 1))
                    report[1].append(8)
                    report[0].append('|')
                    report[1].append(3)
                    report[0].append("Iter:{}/{}/{}".format(k['iter'] + 1, int(iter + 1),n_iter))
                    report[1].append(25)
                    report[0].append('|')
                    report[1].append(1)
                else:
                    report[0].append("Iter:{}/{}".format(int(iter + 1),n_iter))
                    report[1].append(18)
                    report[0].append('|')
                    report[1].append(1)
                report[0].append(
                    "Time:{}/{}/{}".format(printer.timeformatter(timeit.default_timer() - self.iter_start_time),
                    printer.timeformatter(timeit.default_timer() - self.report_start_time), printer.timeformatter(
                    timeit.default_timer() - self.start_time+self.pre_time[0])))
                report[1].append(26)
                report[0].append('|')
                report[1].append(1)
                report[0].append("Loss:%.4f"%(k['train_cost']))
                report[1].append(15)
                report[0].append('|')
                report[1].append(1)
                report[0].append("[%d%%]"%process)
                report[1].append(8)
                iterreport=printer.lineformatter(report[0],LengthList=report[1],Align='center')
                self.logger(iterreport, 1)

    def before_epoch(self):
        if self.kwargs['n_part']==0:
            self.epoch_start_time = timeit.default_timer()



    def after_epoch(self):
        kwargs = self.kwargs
        epoch_time = timeit.default_timer() - self.epoch_start_time
        if self.report_epoch:
            len=24
            self.logger("", 2)
            self.logger("Epoch Done:", 1)
            self.logger("="*24, 1)
            self.logger(printer.lineformatter(["Epoch:%d  " % kwargs['n_epoch']],len,Align='center',FirstColumnLeft=False), 1)
            self.logger(printer.lineformatter(["Update:%d" % (kwargs['n_iter'])],len,Align='center',FirstColumnLeft=False), 1)
            self.logger(printer.lineformatter(["Time Used:%s" % printer.timeformatter(epoch_time)],len,Align='center',FirstColumnLeft=False), 1)
            self.logger(printer.lineformatter(["Loss:%.4f" % kwargs['costs'][-1]],len,Align='center',FirstColumnLeft=False), 1)
            self.logger(printer.lineformatter(["Error:%.4f%%" % (kwargs['test_error'] * 100)],len,Align='center',FirstColumnLeft=False), 1)
            self.logger("="*24, 1)
            self.logger("", 2)
        if self.plot:
            p = OrderedDict()
            for key, param in kwargs['model'].params.items():
                p[key] = param.get_value()
            plt.plot(self.kwargs['costs'], self.kwargs['errors'], p, kwargs['model'].roles)

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
        if 'best_iter' in kwargs and kwargs['best_iter'] != -1:
            self.logger("Best iteration:%s" % kwargs['best_iter'], 1)
        self.logger("Time used in total:%s" % printer.timeformatter((timeit.default_timer() - self.start_time)+self.pre_time[0]), 1)
        self.logger("Finall cost:%.4f" % test_cost, 1)
        if 'best_valid_error' in kwargs and kwargs['best_valid_error'] != 1.:
            self.logger("Best valid error:%.4f%%" % (kwargs['best_valid_error'] * 100), 1)
        self.logger("Finall test error:%.4f%%" % (test_error), 1)

    def save_(self,dict):
        dict['monitor']={'time':self.pre_time[0]+(timeit.default_timer()-self.start_time),'epoch_time':self.pre_time[1]+(timeit.default_timer()-self.epoch_start_time)}
    def load_(self,dict):
        try:
            self.pre_time=[dict['monitor']['time'],dict['monitor']['epoch_time']]
        except:
            pass



config = ex({})
