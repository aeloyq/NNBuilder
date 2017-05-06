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
        self.overwrite=True
        self.load_file_name=''
    def init(self):
        base.init(self)
    def before_train(self):
        kwargs = self.kwargs
        model=kwargs['dim_model']
        layers=model.layers
        path = './%s/save' % (kwargs['conf'].name)
        self.path= './%s/save' % (kwargs['conf'].name)
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
            params=(np.load(path+'/'+self.load_file_name))['save'].tolist()
        #file.close()
            for key in layers:
                for param,sparam in zip(layers[key].params,params[key]):
                    layers[key].params[param].set_value(params[key][sparam])
            kwargs['epoches']=params['epoches']#TODO:may smaller than real number
            kwargs['iteration_total']= params['iteration_total']
            kwargs['best_iter'] = params['best_iter']
            kwargs['best_valid_error'] =params['best_valid_error']
            kwargs['idx'] = params['idx']+1
            for i in params['errors']:
                kwargs['errors'].append(i)
            for i in params['costs']:
                kwargs['costs'].append(i)
            name = ''
            try:
                for ex in kwargs['extension']:
                    name=ex.__class__
                    ex.load_(params)
            except:
                self.logger('{} Load failed'.format(name), 3)
            self.logger('Load sucessfully',3)

    def after_iteration(self):
        kwargs = self.kwargs
        if kwargs['iteration_total'] % self.save_freq == 0:
            savename=self.path + '/%s.npz' % (time.asctime().replace(' ','-').replace(':','_'))
            self.save_npz(savename,self.overwrite)

    def after_train(self):
        kwargs = self.kwargs
        self.save_npz(self.path+'/finall',self.overwrite)

    def save_npz(self,name,delete=True):
        kwargs = self.kwargs
        self.logger("Prepare to save model", 2, 1)
        model = kwargs['dim_model']
        layers = model.layers
        params = OrderedDict()
        params2save = OrderedDict()
        for key in layers:
            params[key] = layers[key].params
        for key in params:
            params2save[key] = OrderedDict()
            for pname, param in params[key].items():
                params2save[key][pname]=param.get_value()
        params2save['epoches'] = kwargs['epoches']
        params2save['iteration_total'] = kwargs['iteration_total']
        params2save['best_iter'] = kwargs['best_iter']
        params2save['best_valid_error'] = kwargs['best_valid_error']
        params2save['errors'] = kwargs['errors']
        params2save['costs'] = kwargs['costs']
        params2save['idx'] = kwargs['idx']
        for ex in kwargs['extension']:
            ex.save_(params2save)
        savename = name
        self.logger("Prepare to save model", 3)
        np.savez(savename, save=params2save)
        self.logger("Save sucessfully at file:{}".format(savename), 3)
        if delete:
            savelist = [name for name in os.listdir(self.path) if name.endswith('.npz')]
            savelist.sort()
            for i in range(len(savelist) - self.save_len):
                os.remove(self.path + '/' + savelist[i])
                self.logger("Deleted old file:{}".format(self.path + '/' + savelist[i]), 3)
config=ex({})

