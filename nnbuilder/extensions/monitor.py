# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import extension
import timeit
import numpy as np
import  nnbuilder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import os
import threading
import theano.d3viz as d3v
from nnbuilder.layers.layers import roles

base=extension.extension
class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.report_iter=False
        self.report_iter_frequence=5
        self.report_epoch = True
        self.plot=False
        self.plot_frequence=-1
        self.dotprint=True
        self.start_iter=-1
    def init(self):
        base.init(self)
        if self.plot_frequence==-1: self.plot_frequence=len(self.kwargs['minibatches'][0])
    def before_train(self):
        kwargs=self.kwargs
        self.epoch_s_time = timeit.default_timer()
        self.iteration_s_time = timeit.default_timer()
        self.start_time=timeit.default_timer()
        self.path='./{}/tmp/'.format(nnbuilder.config.name)
        if not os.path.exists('./{}'.format(nnbuilder.config.name)):os.mkdir('./{}'.format(nnbuilder.config.name))
        if not os.path.exists('./{}/tmp'.format(nnbuilder.config.name)): os.mkdir('./{}/tmp'.format(nnbuilder.config.name))

        self.params = kwargs['dim_model'].params
        self.roles = kwargs['dim_model'].roles
        if self.start_iter==-1:
            self.start_iter=0
            self.plot_costs = []
            self.plot_errors = []
        if self.dotprint:d3v.d3viz(kwargs['dim_model'].output.output,self.path + 'model.html')
    def after_iteration(self):
        iteration_time=timeit.default_timer()-self.iteration_s_time
        self.iteration_s_time=timeit.default_timer()
        if self.report_iter:

            iter = self.kwargs['iteration_total']
            if iter % self.report_iter_frequence == 0:
                self.logger("Iteration Report at Epoch:%d   Iteration:%d   Time Used:%.2fs   " \
                            "Cost:%.4f" % (self.kwargs['epoches'], iter,
                                           iteration_time, self.kwargs['train_result']), 2)

        if self.plot:

            iter = self.kwargs['iteration_total']
            if iter % (self.plot_frequence) == 0:
                train_result=self.kwargs['train_result']
                errors=self.kwargs['errors']
                t_plot=threading.Thread(target=self.plot_func,name='monitor.plot',args=(iter,train_result,errors))
                t_plot.start()


    def plot_func(self,iter,train_result,errors):
        x_axis = range(self.start_iter, iter, self.plot_frequence)

        if len(x_axis) > len(self.plot_costs):
            self.plot_costs.append(train_result)
            if len(errors) > 0:
                self.plot_errors.append(errors[-1])
            else:
                self.plot_errors.append(1.)

        plt.figure(1)
        plt.cla()
        x_axis=range(self.start_iter,iter,self.plot_frequence)
        plt.title(nnbuilder.config.name)
        plt.ylabel('Loss')
        plt.xlabel('Iters')
        plt.plot(x_axis,self.plot_costs,label='Loss',color='orange')
        plt.legend()
        plt.savefig(self.path+'process_cost.png')

        plt.figure(2)
        plt.cla()
        x_axis = range(self.start_iter, iter, self.plot_frequence)
        plt.title(nnbuilder.config.name)
        plt.ylabel('Error')
        plt.xlabel('Iters')
        plt.plot(x_axis, self.plot_errors, label='Error', color='blue')
        plt.legend()
        plt.savefig(self.path + 'process_error.png')

        n_im = len(self.params)
        a = np.int(np.sqrt(n_im))
        b = a
        if a * b < n_im: a += 1
        if a * b < n_im: b += 1
        plt.figure(3, (b*4,a*4))
        plt.cla()

        i=0
        for key,param in self.params.items():
            i += 1
            if self.roles[key] is roles.weight:
                plt.subplot(a,b,i)
                value=param.get_value()
                plt.title(key+' '+str(value.shape))
                img=np.asarray(value)
                if img.ndim!=1:
                    plt.imshow(img,cmap='gray')
            else:
                plt.subplot(a, b, i)
                y=param.get_value()
                plt.title(key+' '+str(y.shape))
                x_axis_bi=np.arange(y.shape[0])
                plt.plot(x_axis_bi,y,color='black')
        plt.savefig(self.path + 'paramsplot.png')




    def after_epoch(self):
        kwargs=self.kwargs
        epoch_time = timeit.default_timer() - self.epoch_s_time
        self.epoch_s_time = timeit.default_timer()
        if self.report_epoch:
            self.logger("", 2)
            self.logger( "◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆◆",2)
            self.logger( "  Single Epoch Done:",2)
            self.logger( "  Epoches:%d  " % kwargs['epoches'],2)
            self.logger( "  Iterations:%d" % (kwargs['iteration_total']),2)
            self.logger( "  Time Used:%.2fs" % epoch_time,2)
            self.logger( "  Cost:%.4f   " % kwargs['costs'][-1],2)
            self.logger( "  Error:%.4f%%" % (kwargs['train_error']*100),2)
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
        test_result = np.array([test_model(*tuple(testdata)) for testdata in testdatas])
        test_error = np.mean(test_result[:, 1])
        test_cost=np.mean(test_result[:, 0])
        self.logger("", 0)
        self.logger("All Finished:",0)
        self.logger("Trainning finished after epoch:%s"%kwargs['epoches'],1)
        self.logger("Trainning finished at iteration:%s"%kwargs['iteration_total'],1)
        self.logger("Best iteration:%s"% kwargs['best_iter'],1)
        self.logger("Time used in total:%.2fs"%total_time,1)
        self.logger("Finall cost:%.4f"% test_cost,1)
        self.logger("Best valid error:%.4f%%" % (kwargs['best_valid_error'] * 100),1)
        self.logger("Finall test error:%.4f%%" % (test_error * 100),1)


    def save_(self,dict):
        kwargs = self.kwargs
        x_axis = range(self.start_iter, kwargs['iteration_total'], self.plot_frequence)
        if len(x_axis) > len(self.plot_costs):
            self.plot_costs.append(kwargs['train_result'])
            if len(kwargs['errors']) > 0:
                self.plot_errors.append(kwargs['errors'][-1])
            else:
                self.plot_errors.append(1.)
        dict['start_iter']= self.start_iter
        dict['plot_costs']= self.plot_costs
        dict['plot_errors']= self.plot_errors
    def load_(self,dict):
        self.start_iter=dict['start_iter']
        self.plot_costs=dict['plot_costs']
        self.plot_errors=dict['plot_errors']

config=ex({})

