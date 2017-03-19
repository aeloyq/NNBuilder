# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:45:34 2016

@author: aeloyq
"""
import extension
import time
import numpy as np
import os
from collections import OrderedDict

base=extension.extension
class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.save_freq=10000
        self.save_len=3
        self.load=True
        self.save=True
        self.load_file_name=''
    def init(self):
        base.init(self)
    def before_train(self):
        kwargs = self.kwargs
        model=kwargs['dim_model']
        layers=model.layers
        path = './%s/save' % (kwargs['conf'].name)
        if self.load:
            self.logger('Loading saved model from checkpoint:',2,1)
            if self.load_file_name=='':
                savelist = [name for name in os.listdir(path) if name.endswith('.npz')]
                if savelist==[]:
                    self.logger('Checkpoint not found, exit loading', 3)
                    return
                savelist.sort()
                self.load_file_name=savelist[-1]
            self.logger('Checkpoint is found:{}...'.format(self.load_file_name), 3)
            #file=open(path+'/'+self.load_file_name,'rb')
            params=np.load(path+'/'+self.load_file_name)
        #file.close()
            for key in layers:
                for param,sparam in zip(layers[key].params,params[key]):
                    param.set_value(sparam)
            kwargs['epoch_time'][0] = params['epoch_time'][0]
            kwargs['epoches'][0]=params['epoches'][0]#TODO:may smaller than real number
            kwargs['iteration_total'][0] = params['iteration_total'][0]
            kwargs['best_iter'][0] = params['best_iter'][0]
            kwargs['best_valid_error'][0] =params['best_valid_error'][0]
            for i in params['errors']:
                kwargs['errors'].append(i)
            for i in params['costs']:
                kwargs['costs'].append(i)
            self.logger('Load sucessfully',3)

    def after_iteration(self):
        kwargs = self.kwargs
        if kwargs['iteration_total'][0] % self.save_freq == 0:
            self.logger("Prepare to save model",2,1)
            model = kwargs['dim_model']
            layers = model.layers
            params = OrderedDict()
            params2save=OrderedDict()
            path = './%s/save' % (kwargs['conf'].name)
            for key in layers:
                params[key] = layers[key].params
            for key in params:
                params2save[key]=[]
                for idx, param in enumerate(params[key]):
                    params2save[key].append(param.get_value())
            params2save['epoch_time']=kwargs['epoch_time']
            params2save['epoches'] = kwargs['epoches']
            params2save['iteration_total'] = kwargs['iteration_total']
            params2save['best_iter'] = kwargs['best_iter']
            params2save['best_valid_error'] = kwargs['best_valid_error']
            params2save['errors'] = kwargs['errors']
            params2save['costs'] = kwargs['costs']
            #file=open(path + '/%s.npz' % (time.asctime().replace(' ','-').replace(':','_')),'wb')
            savename=path + '/%s.npz' % (time.asctime().replace(' ','-').replace(':','_'))
            self.logger("Prepare to save model", 3)
            np.savez(savename, **params2save)
            self.logger("Save sucessfully at file:{}".format(savename), 3)
            #file.close()
            savelist = [name for name in os.listdir(path) if name.endswith('.npz')]
            savelist.sort()
            for i in range(len(savelist) - self.save_len):
                os.remove(path + '/' + savelist[0])
                self.logger("Deleted old file:{}".format(path + '/' + savelist[0]), 3)
    def after_train(self):
        kwargs = self.kwargs
        model = kwargs['dim_model']
        layers = model.layers
        params = OrderedDict()
        params2save=OrderedDict()
        path='./%s/save'%(kwargs['conf'].name)
        for key in layers:
            params[key] = layers[key].params
        for key in params:
            params2save[key] = []
            for idx, param in enumerate(params[key]):
                params2save[key].append(param.get_value())
        params2save['epoch_time'] = kwargs['epoch_time']
        params2save['epoches'] = kwargs['epoches']
        params2save['iteration_total'] = kwargs['iteration_total']
        params2save['best_iter'] = kwargs['best_iter']
        params2save['best_valid_error'] = kwargs['best_valid_error']
        params2save['errors'] = kwargs['errors']
        params2save['costs'] = kwargs['costs']
        #file=open(path+'finall_model.npz','wb')
        np.savez(file,**params2save)
        #file.close()
config=ex({})

