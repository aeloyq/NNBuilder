# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import timeit
import numpy as np
import nnbuilder
import os
import saveload
import earlystop
import thread
from basic import base
from nnbuilder.layers.roles import *
from collections import OrderedDict
from nnbuilder.kernel import *
from nnbuilder.main import mainloop
from nnbuilder.tools import printer, plotter


class ex(base):
    def __init__(self, kwargs):
        base.__init__(self, kwargs)
        self.report_iter = False
        self.report_iter_frequence = 5
        self.report_epoch = True
        self.report_epoch_frequence = 1
        self.compare_repo = None
        self.plot = False
        self.dotprint = True

    def init(self):
        base.init(self)
        self.path = './{}/plot/'.format(nnbuilder.config.name)
        self.batches = int(str(np.sum(self.kwargs['n_data'][1])))
        self.pre_time = [0, 0]
        self.compare_saving = []
        self.compare_name = []

        def load_data(file):
            return np.load(file)['mainloop'].tolist()

        if self.compare_repo is not None and self.compare_repo != []:
            if isinstance(self.compare_repo, (tuple, list)):
                for repo in self.compare_repo:
                    assert isinstance(repo, str)
                    savefile = './{}/save/final/final.npz'.format(repo)
                    if os.path.isfile(savefile):
                        self.compare_saving.append(load_data(savefile))
                        self.compare_name.append(repo)
                    else:
                        path = './{}/save'.format(repo)
                        savelist = [path + name for name in os.listdir(path) if name.endswith('.npz')]
                        assert savelist > 0
                        savelist.sort(saveload.ex.compare_savefiles)
                        self.compare_saving.append(load_data(savelist[-1]))
            else:
                savelist = [self.compare_repo + name for name in os.listdir(self.compare_repo) if name.endswith('.npz')]
                for filename in savelist:
                    self.compare_saving.append(load_data(self.compare_repo + '/' + filename))
                    self.compare_name.append(filename)

    def before_train(self):
        self.start_time = timeit.default_timer()
        self.epoch_start_time = timeit.default_timer()
        self.iter_start_time = timeit.default_timer()
        self.report_start_time = timeit.default_timer()

        if not os.path.exists('./{}'.format(nnbuilder.config.name)): os.mkdir('./{}'.format(nnbuilder.config.name))
        if not os.path.exists('./{}/plot/model'.format(nnbuilder.config.name)): os.mkdir(
            './{}/plot/model'.format(nnbuilder.config.name))
        if self.dotprint: kernel.printing(self.kwargs['model'].raw_output, self.path + 'model/' + 'model.html')

    def before_init_iter(self):
        self.init_iter_start_time = timeit.default_timer()

    def after_init_iter(self):
        kwargs = self.kwargs
        init_iter_time = timeit.default_timer() - self.init_iter_start_time
        len = 24
        self.logger("", 2)
        self.logger("=" * 24, 1)
        self.logger(
            printer.lineformatter(["Initate Test"], len, Align='center', FirstColumnLeft=False), 1)
        self.logger(printer.lineformatter(["Time Used:%s" % printer.timeformatter(init_iter_time)], len, Align='center',
                                          FirstColumnLeft=False), 1)
        self.logger(
            printer.lineformatter(["Loss:%.4f" % kwargs['losses'][-1]], len, Align='center', FirstColumnLeft=False), 1)
        self.logger(printer.lineformatter(["Error:%.4f%%" % (kwargs['test_error'] * 100)], len, Align='center',
                                          FirstColumnLeft=False), 1)
        self.logger("=" * 24, 1)
        self.logger("", 2)

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
                    report[0].append("Iter:{}/{}/{}".format(k['iter'] + 1, int(iter + 1), n_iter))
                    report[1].append(25)
                    report[0].append('|')
                    report[1].append(1)
                else:
                    report[0].append("Iter:{}/{}".format(int(iter + 1), n_iter))
                    report[1].append(18)
                    report[0].append('|')
                    report[1].append(1)
                report[0].append(
                    "Time:{}/{}/{}".format(printer.timeformatter(timeit.default_timer() - self.iter_start_time),
                                           printer.timeformatter(timeit.default_timer() - self.report_start_time),
                                           printer.timeformatter(
                                               timeit.default_timer() - self.start_time + self.pre_time[0])))
                report[1].append(26)
                report[0].append('|')
                report[1].append(1)
                report[0].append("Loss:%.4f" % (k['train_loss']))
                report[1].append(15)
                report[0].append('|')
                report[1].append(1)
                report[0].append("[%d%%]" % process)
                report[1].append(8)
                iterreport = printer.lineformatter(report[0], LengthList=report[1], Align='center')
                self.logger(iterreport, 1)

    def before_epoch(self):
        if self.kwargs['n_part'] == 0:
            self.epoch_start_time = timeit.default_timer()

    def after_epoch(self):
        kwargs = self.kwargs
        epoch_time = timeit.default_timer() - self.epoch_start_time + self.pre_time[1]
        self.pre_time[1] = 0
        if self.report_epoch:
            if self.kwargs['n_epoch'] % self.report_epoch_frequence == 0:
                len = 24
                self.logger("", 2)
                self.logger("=" * 24, 1)
                self.logger(
                    printer.lineformatter(["Epoch:%d  " % kwargs['n_epoch']], len, Align='center',
                                          FirstColumnLeft=False),
                    1)
                self.logger(
                    printer.lineformatter(["Update:%d" % (kwargs['n_iter'])], len, Align='center',
                                          FirstColumnLeft=False),
                    1)
                self.logger(
                    printer.lineformatter(["Time Used:%s" % printer.timeformatter(epoch_time)], len, Align='center',
                                          FirstColumnLeft=False), 1)
                self.logger(
                    printer.lineformatter(["Loss:%.4f" % kwargs['losses'][-1]], len, Align='center',
                                          FirstColumnLeft=False),
                    1)
                self.logger(printer.lineformatter(["Error:%.4f%%" % (kwargs['test_error'] * 100)], len, Align='center',
                                                  FirstColumnLeft=False), 1)
                self.logger("=" * 24, 1)
                self.logger("", 2)
        if self.plot:
            valid_progress = None
            if earlystop in self.kwargs['extensions']:
                valid_progress=[earlystop.instance.valid_losses,earlystop.instance.valid_errors]
            thread.start_new_thread(plotter.monitor_progress, (nnbuilder.config.name,
                                                               self.path + 'progress/progress.html', self.kwargs['losses'],
                                                               self.kwargs['errors'], self.compare_saving,
                                                               self.compare_name, valid_progress))

    def after_train(self):
        kwargs = self.kwargs
        test_minibatches = kwargs['minibatches'][2]
        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['datas']
        test_fn = kwargs['test_fn']
        testdatas = []
        for index in test_minibatches:
            data = mainloop.prepare_data(test_X, test_Y, index, kwargs['model'])
            testdatas.append(data)
        test_result = np.array([test_fn(*tuple(testdata)) for testdata in testdatas])
        test_error = float(str(np.mean(test_result[:, 1]) * 100))
        test_loss = float(str(np.mean(test_result[:, 0])))
        self.logger("", 0)
        self.logger("All Finished:", 0)
        self.logger("Trainning Finished At Epoch:%s   Iter:%s" % (kwargs['n_epoch'], kwargs['n_iter']), 1)
        self.logger("Best Loss:%.4f   At Epoch:%s" % (np.min(kwargs['losses']), np.argmin(kwargs['losses'])), 1)
        self.logger("Best Error:%.2f%%   At Epoch:%s" % (np.min(kwargs['errors']) * 100, np.argmin(kwargs['errors'])),
                    1)
        self.logger("Time Used In Total:%s" % printer.timeformatter(
            (timeit.default_timer() - self.start_time) + self.pre_time[0]), 1)
        self.logger("Finall Loss:%.4f" % test_loss, 1)
        self.logger("Finall Test Error:%.4f%%" % (test_error), 1)

    def save_(self, dict):
        dict = {'time': self.pre_time[0] + (timeit.default_timer() - self.start_time),
                'epoch_time': self.pre_time[1] + (timeit.default_timer() - self.epoch_start_time)}
        return dict

    def load_(self, dict):
        try:
            self.pre_time = [dict['time'], dict['epoch_time']]
        except:
            self.logger("Didn't use monitor in latest training Abort monitor loading", 2)


config = ex({})
instance = config
