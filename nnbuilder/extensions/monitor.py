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
from collections import OrderedDict

base=extension.extension
class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.report_iter=False
        self.report_iter_frequence=5
        self.report_epoch = True
        self.plot=True
        self.plot_frequence=1000
        self.start_iter=-1
    def init(self):
        base.init(self)
    def before_train(self):
        kwargs=self.kwargs
        self.epoch_s_time = timeit.default_timer()
        self.iteration_s_time = timeit.default_timer()
        self.start_time=timeit.default_timer()
        self.path='./{}/tmp/'.format(nnbuilder.config.name)
        if not os.path.exists('./{}'.format(nnbuilder.config.name)):os.mkdir('./{}'.format(nnbuilder.config.name))
        if not os.path.exists('./{}/tmp'.format(nnbuilder.config.name)): os.mkdir('./{}/tmp'.format(nnbuilder.config.name))

        self.params = OrderedDict()
        for key in kwargs['dim_model'].layers:
            for param in kwargs['dim_model'].layers[key].params:
                self.params[param.name] = param
        if self.start_iter==-1:
            self.start_iter=0
            self.plot_costs = []
            self.plot_errors = []
        d3v.d3viz(kwargs['dim_model'].output.output,self.path + 'model.html')
    def after_iteration(self):
        kwargs = self.kwargs
        iteration_time=timeit.default_timer()-self.iteration_s_time
        self.iteration_s_time=timeit.default_timer()



    def plot_func(self,kwargs):
        iter = kwargs['iteration_total']

        if iter%(self.plot_frequence)==0:

            x_axis = range(self.start_iter, iter, self.plot_frequence)

            if len(x_axis) > len(self.plot_costs):
                self.plot_costs.append(kwargs['train_result'])
                if len(kwargs['errors']) > 0:
                    self.plot_errors.append(kwargs['errors'][-1])
                else:
                    self.plot_errors.append(1.)

            plt.figure(1)
            plt.cla()
            x_axis=range(self.start_iter,iter,self.plot_frequence)
            plt.title(nnbuilder.config.name)
            plt.ylabel('Loss/Error')
            plt.xlabel('Iters')
            plot_costs=self.plot_costs
            if max(plot_costs)>1:
                plot_costs=(np.array(plot_costs)/max(plot_costs)).tolist()
            plt.plot(x_axis,plot_costs,label='Loss',color='orange')
            plt.plot(x_axis, self.plot_errors, label='Error', color='blue')
            plt.legend()
            plt.savefig(self.path+'process.png')

            n_im = len(self.params)
            a = np.int(np.sqrt(n_im))
            b = a
            if a * b < n_im: a += 1
            if a * b < n_im: b += 1
            plt.figure(2, (b*4,a*4))
            plt.cla()

            i=0
            for key in self.params:
                i += 1
                if key.find('Bi')==-1:
                    plt.subplot(a,b,i)
                    plt.title(key)
                    img=np.asarray(self.params[key].get_value())
                    img=(img-np.min(img))
                    img=img/np.max(img)
                    img=img*255
                    if img.ndim!=1:
                        plt.imshow(img,cmap='gray')
                else:
                    plt.subplot(a, b, i)
                    plt.title(key)
                    y=self.params[key].get_value()
                    x_axis_bi=range(y.shape[0])
                    y=y+np.min(y)
                    y=(y*2)/np.max(y)-1
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
        test_error = np.mean([test_model(*tuple(testdata)) for testdata in testdatas])
        self.logger("", 0)
        self.logger("All Finished:",0)
        self.logger("Trainning finished after epoch:%s"%kwargs['epoches'],1)
        self.logger("Trainning finished at iteration:%s"%kwargs['iteration_total'],1)
        self.logger("Best iteration:%s"% kwargs['best_iter'],1)
        self.logger("Time used in total:%.2fs"%total_time,1)
        self.logger("Finall cost:%s"% kwargs['costs'][-1],1)
        self.logger("Finall error:%.4f%%" % (kwargs['errors'][-1] * 100),1)
        self.logger("Test error:%.4f%%" % (test_error * 100),1)
        self.logger("Best error:%.4f%%" % (kwargs['best_valid_error'] * 100),1)


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

