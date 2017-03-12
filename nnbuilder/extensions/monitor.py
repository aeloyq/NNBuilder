# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import extension
import timeit
import numpy as np

base=extension.extension
class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.report_iter=False
        self.report_epoch = True
    def init(self):
        base.init(self)
    def before_train(self):
        self.epoch_s_time = timeit.default_timer()
        self.iteration_s_time = timeit.default_timer()
        self.start_time=timeit.default_timer()
    def after_iteration(self):
        kwargs = self.kwargs
        iteration_time=timeit.default_timer()-self.iteration_s_time
        self.iteration_s_time=timeit.default_timer()
        if self.report_iter:
            self.logger( "Iteration Report at Epoch:%d   Iteration:%d   Time Used:%.2fs   " \
                  "Cost:%.4f" % (kwargs['epoches'][0], kwargs['iteration_total'][0],
                                 iteration_time,kwargs[' train_cost'][0]),2)
    def after_epoch(self):
        kwargs=self.kwargs
        epoch_time = timeit.default_timer() - self.epoch_s_time
        self.epoch_s_time = timeit.default_timer()
        if self.report_epoch:
            self.logger("", 2)
            self.logger( "◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆",2)
            self.logger( "  Single Epoch Done:",2)
            self.logger( "  Epoches:%d  " % kwargs['epoches'][0],2)
            self.logger( "  Iterations:%d" % (kwargs['iteration_total'][0]),2)
            self.logger( "  Time Used:%.2fs" % epoch_time,2)
            self.logger( "  Cost:%.4f   " % kwargs['costs'][-1],2)
            self.logger( "  Error:%.4f%%" % (kwargs['train_error'][0]*100),2)
            self.logger( "◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆",2)
            self.logger("", 2)
    def after_train(self):
        kwargs = self.kwargs
        test_minibatches=kwargs['minibatches'][2]
        total_time=timeit.default_timer()-self.start_time
        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['data_stream']
        test_model=kwargs['test_model']
        testdatas = []
        for _, index in test_minibatches:
            data = kwargs['prepare_data'](test_X, test_Y, index)
            testdatas.append(data)
        test_error = np.mean([test_model(*tuple(testdata)) for testdata in testdatas])
        self.logger("", 0)
        self.logger("All Finished:",0)
        self.logger("Trainning finished after epoch:%s"%kwargs['epoches'][0],1)
        self.logger("Trainning finished at iteration:%s"%kwargs['iteration_total'][0],1)
        self.logger("Best iteration:%s"% kwargs['best_iter'][0],1)
        self.logger("Time used in total:%.2fs"%total_time,1)
        self.logger("Finall cost:%s"% kwargs['costs'][-1],1)
        self.logger("Finall error:%.4f%%" % (kwargs['errors'][-1] * 100),1)
        self.logger("Test error:%.4f%%" % (test_error * 100),1)
        self.logger("Best error:%.4f%%" % (kwargs['best_valid_error'][0] * 100),1)

config=ex({})

