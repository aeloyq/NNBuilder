# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import numpy as np
import theano
import theano.tensor as t

a=t.matrix('a')
b=t.matrix('b')


o,u=theano.scan()

'''
       if self.report_iter:

           iter=kwargs['iteration_total']
           if iter%self.report_iter_frequence==0:
               self.logger( "Iteration Report at Epoch:%d   Iteration:%d   Time Used:%.2fs   " \
                     "Cost:%.4f" % (kwargs['epoches'], iter,
                                    iteration_time,kwargs['train_result']),2)

       if self.plot:
           pass
           #t_plot=threading.Thread(target=self.plot_func,name='monitor.plot',args=(kwargs,))
           #t_plot.start()'''
