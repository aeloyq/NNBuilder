# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import timeit
from basic import *
from nnbuilder.tools import printer


class Monitor(ExtensionBase):
    def init(self):
        '''
        verbose:
            0-1:no log output just plot
            2-4:epoch only and use line-layout
            >=5:epoch and log iter per 1/n of epoch
        :param kwargs:
        '''
        self.n_batch = (self.data.size - 1) // self.config.batch_size + 1
        self.pre_time = [0, 0]

    def before_train(self):
        self.start_time = timeit.default_timer()
        self.epoch_start_time = timeit.default_timer()
        self.iter_start_time = timeit.default_timer()
        self.report_start_time = timeit.default_timer()

    def before_init_iter(self):
        self.init_iter_start_time = timeit.default_timer()

    def after_init_iter(self):
        kwargs = self.train_history
        init_iter_time = timeit.default_timer() - self.init_iter_start_time
        if self.is_log_detail():
            len = 24
            self.logger("", 2)
            self.logger("=" * 24, 1)
            self.logger(
                printer.lineformatter(["Initate Test"], len, Align='center', FirstColumnLeft=False), 1)
            self.logger(
                printer.lineformatter(["Time Used:%s" % printer.timeformatter(init_iter_time)], len, Align='center',
                                      FirstColumnLeft=False), 1)
            self.logger(
                printer.lineformatter(["Loss:%.4f" % kwargs['losses'][-1]], len, Align='center', FirstColumnLeft=False),
                1)
            for m in self.model.metrics:
                self.logger(
                    printer.lineformatter([m.name + ":%.4f" % (kwargs['scores'][m.name][-1])], len, Align='center',
                                          FirstColumnLeft=False), 1)
            self.logger("=" * 24, 1)
            self.logger("", 2)
        else:
            k = self.train_history
            report = [[], []]
            report[0].append("Initate  Test")
            report[1].append(15)
            report[0].append("Loss:%.4f" % (k['losses'][-1]))
            report[1].append(15)
            for m in k['model'].metrics:
                report[0].append(m.name + ':%.2f' % k['scores'][m.name][-1])
                report[1].append(12)
            self.logger(printer.lineformatter(report[0], LengthList=report[1], Align='center'), 1)

    def is_plot_iter(self):
        verbose = self.get_verbose()
        iter = self.train_history['iter']
        return (iter + 1 in [int(float(self.n_batch) * i / verbose) for i in
                             range(1, verbose + 1)] and verbose >= 5) or ((iter + 1) % (
            -verbose) == 0 and verbose < 0)

    def before_iteration(self):
        self.iter_start_time = timeit.default_timer()
        if self.is_log_detail():
            if self.is_plot_iter():
                self.report_start_time = timeit.default_timer()

    def after_iteration(self):
        if self.is_log_detail():
            n_iter = self.train_history['n_iter']
            iter = self.train_history['iter']
            progress = ((iter + 1) * 100) / self.n_batch
            if self.is_plot_iter():
                k = self.train_history
                report = [[], []]
                report[0].append("Epoch:{}".format(k['n_epoch'] + 1))
                report[1].append(10)
                report[0].append('|')
                report[1].append(1)
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
                report[0].append("[%d%%]" % progress)
                report[1].append(8)
                iterreport = printer.lineformatter(report[0], LengthList=report[1], Align='center')
                self.logger(iterreport, 1)

    def before_epoch(self):
        self.epoch_start_time = timeit.default_timer()

    def after_epoch(self):
        verbose = self.get_verbose()
        kwargs = self.train_history
        epoch_time = timeit.default_timer() - self.epoch_start_time + self.pre_time[1]
        self.pre_time[1] = 0
        if self.is_log_inline():
            freq = [None, None, 100, 10, 1][verbose]
        else:
            freq = 1
        if not self.is_log_silent():
            if self.train_history['n_epoch'] % freq == 0:
                if freq == 1 and self.is_log_detail():
                    length = 24
                    self.logger("", 1)
                    self.logger("=" * 24, 1)
                    self.logger(
                        printer.lineformatter(["Epoch:%d  " % kwargs['n_epoch']], length, Align='center',
                                              FirstColumnLeft=False),
                        1)
                    self.logger(
                        printer.lineformatter(["Iter:%d" % (kwargs['n_iter'])], length, Align='center',
                                              FirstColumnLeft=False),
                        1)
                    self.logger(
                        printer.lineformatter(["Time Used:%s" % printer.timeformatter(epoch_time)], length,
                                              Align='center',
                                              FirstColumnLeft=False), 1)
                    self.logger(
                        printer.lineformatter(["Loss:%.4f" % kwargs['losses'][-1]], length, Align='center',
                                              FirstColumnLeft=False),
                        1)
                    for m in self.model.metrics:
                        self.logger(
                            printer.lineformatter([m.name + ":%.4f" % (kwargs['scores'][m.name][-1])], length,
                                                  Align='center', FirstColumnLeft=False), 1)
                    self.logger("=" * 24, 1)
                    self.logger("", 1)
                else:
                    k = self.train_history
                    report = [[], []]
                    report[0].append("Epoch:{}".format(k['n_epoch']))
                    report[1].append(10)
                    report[0].append('|')
                    report[1].append(1)
                    report[0].append("Iter:{}".format(k['n_iter']))
                    report[1].append(18)
                    report[0].append('|')
                    report[1].append(1)
                    report[0].append("Time:{}".format(printer.timeformatter(epoch_time)))
                    report[1].append(26)
                    report[0].append('|')
                    report[1].append(1)
                    report[0].append("Loss:%.4f" % (k['losses'][-1]))
                    report[1].append(15)
                    report[0].append('|')
                    report[1].append(1)
                    for m in k['model'].metrics:
                        report[0].append(m.name + ':%.2f' % k['scores'][m.name][-1])
                        report[1].append(len(report[0][-1]) + 4)
                    self.logger(printer.lineformatter(report[0], LengthList=report[1], Align='center'), 1)

    def after_train(self):
        test_result = self.MainLoop.test(self.model, self.data, {}, False)
        if self.is_log_detail():
            self.logger("", 0)
            self.logger("Last Update At Epoch:%s   Iter:%s" % (self.train_history['n_epoch'], self.train_history['n_iter']), 1)
        self.logger("Time Used:%s" % printer.timeformatter(
            (timeit.default_timer() - self.start_time) + self.pre_time[0]), 1)
        self.logger("Best Loss:%.4f(%s)" % (np.min(self.train_history['losses']), np.argmin(self.train_history['losses'])),
                    1)
        if self.model.metrics[0].down_direction:
            self.logger(
                "Best %s:%.4f(%s)" % (self.model.metrics[0].name,
                                      np.min(self.train_history['scores'][self.model.metrics[0].name]),
                                      np.argmin(self.train_history['scores'][self.model.metrics[0].name])),
                1)
        else:
            self.logger(
                "Best %s:%.4f(%s)" % (self.model.metrics[0].name,
                                      np.max(self.train_history['scores'][self.model.metrics[0].name]),
                                      np.argmax(self.train_history['scores'][self.model.metrics[0].name])),
                1)
        if self.is_log_detail():
            self.logger("Finall Loss:%.4f" % test_result['loss'], 1)
            for m in self.model.metrics:
                self.logger("Finall %s:%.4f" % (m.name, test_result[m.name]), 1)

    def save_(self, dict):
        dict = {'time': self.pre_time[0] + (timeit.default_timer() - self.start_time),
                'epoch_time': self.pre_time[1] + (timeit.default_timer() - self.epoch_start_time)}
        return dict

    def load_(self, dict):
        try:
            self.pre_time = [dict['time'], dict['epoch_time']]
        except:
            self.logger("Didn't use monitor in latest training Abort monitor loading", 2)


monitor = Monitor()
