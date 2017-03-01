# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import Extension
import timeit
import numpy as np

base=Extension.extension
class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.report_iter=False
        self.report_epoch = True
    def before_train(self):
        self.epoch_s_time = timeit.default_timer()
        self.iteration_s_time = timeit.default_timer()
        self.start_time=timeit.default_timer()
    def after_iteration(self):
        kwargs = self.kwargs
        iteration_time=timeit.default_timer()-self.iteration_s_time
        self.iteration_s_time=timeit.default_timer()
        if self.report_iter:
            print "Iteration Report at Epoch:%d   Iteration:%d   Time Used:%.2fs   " \
                  "Cost:%.4f" % (kwargs['epoches'][0], kwargs['iteration_total'][0],
                                 iteration_time,kwargs[' train_cost'][0])
    def after_epoch(self):
        kwargs=self.kwargs
        epoch_time = timeit.default_timer() - self.epoch_s_time
        self.epoch_s_time = timeit.default_timer()
        if self.report_epoch:
            print ''
            print "◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆"
            print "  Single Epoch Done:"
            print "  Epoches:%d  " % kwargs['epoches'][0]
            print "  Iterations:%d" % (kwargs['iteration_total'][0])
            print "  Time Used:%.2fs" % epoch_time
            print "  Cost:%.4f   " % kwargs['costs'][-1]
            print "  Error:%.4f%%" % (kwargs['train_error'][0]*100)
            print "◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆"
            print ''
            for result in kwargs['train_result'][1:]:
                print result
    def after_train(self):
        kwargs = self.kwargs
        test_minibatches=kwargs['minibatches'][2]
        total_time=timeit.default_timer()-self.start_time
        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['data_stream']
        datastream, train_model, valid_model, test_model, sample_model, debug_model, model, NNB_model, optimizer = kwargs['model_stream']
        testdatas = []
        for _, index in test_minibatches:
            data = kwargs['prepare_data'](test_X, test_Y, index)
            testdatas.append(data)
        test_error = np.mean([test_model(*tuple(testdata)) for testdata in testdatas])
        print "Trainning finished after epoch:", kwargs['epoches'][0]
        print "Trainning finished at iteration:", kwargs['iteration_total'][0]
        print "Best iteration:", kwargs['best_iter'][0]
        print "Time used in total:%.2f"%total_time
        print "Finall cost:", kwargs['costs'][-1]
        print "Finall error:%.4f%%" % (kwargs['errors'][-1] * 100)
        print "Test error:%.4f%%" % (test_error * 100)
        print "Best error:%.4f%%" % (kwargs['best_valid_error'][0] * 100)

config=ex({})

