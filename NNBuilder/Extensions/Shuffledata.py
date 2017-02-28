# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import Extension
import timeit

base=Extension.extension
class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.report_iter=False
        self.report_epoch = True
        self.kwargs=kwargs
        self.iteration_total = 0
        self.iteration_time = 0.
        self.epoch_s_time = timeit.default_timer()
        self.iteration_s_time = timeit.default_timer()
    def after_iteration(self):
        kwargs = self.kwargs
        if self.report_iter:
            print "Iteration Report at Epoch:%d   Iteration:%d   Time Used:%.2fs   " \
                  "Cost:%.4f" % (kwargs['epoches'], kwargs['iteration_total'],
                                 kwargs['iteration_time'],kwargs[' train_cost'])
    def after_epoch(self):
        kwargs=self.kwargs
        if self.report_epoch:
            print ''
            print "◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆"
            print "  Single Epoch Done:"
            print "  Epoches:%d  " % kwargs['epoches']
            print "  Iterations:%d" % (kwargs['iteration_total'])
            print "  Time Used:%.2fs" % kwargs['epoch_time']
            print "  Cost:%.4f   " % kwargs['costs'][-1]
            print "  Error:%.4f%%" % (kwargs['train_error']*100)
            print "◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆"
            print ''
            for result in kwargs['train_result'][1:]:
                print result
    def after_train(self):
        kwargs = self.kwargs
        print "Trainning finished after epoch:", kwargs['epoches']
        print "Trainning finished at iteration:", kwargs['iteration_total']
        print "Best iteration:", kwargs['best_iter']
        print "Finall cost:", kwargs['costs'][-1]
        print "Finall error:%.4f%%" % (kwargs['errors'][-1] * 100)
        print "Test error:%.4f%%" % (kwargs['test_error'] * 100)
        print "Best error:%.4f%%" % (kwargs['best_valid_error'] * 100)

