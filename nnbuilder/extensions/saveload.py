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
        if self.load_file_name=='':
            savelist = [name for name in os.listdir(path) if name.endswith('.npz')]
            if savelist==[]:
                return
            savelist.sort()
            self.load_file_name=savelist[-1]
        #file=open(path+'/'+self.load_file_name,'rb')
        params=np.load(path+'/'+self.load_file_name)
        #file.close()
        for key in layers:
            for param,sparam in zip(layers[key].params,params[key]):
                param.set_value(sparam)

    def after_iteration(self):
        kwargs = self.kwargs
        if kwargs['iteration_total'][0] % self.save_freq == 0:
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
                    params2save[key].append( param.get_value())
            #file=open(path + '/%s.npz' % (time.asctime().replace(' ','-').replace(':','_')),'wb')
            np.savez(path + '/%s.npz' % (time.asctime().replace(' ','-').replace(':','_')), **params2save)
            #file.close()
            savelist = [name for name in os.listdir(path) if name.endswith('.npz')]
            savelist.sort()
            for i in range(len(savelist) - self.save_len):
                os.remove(path + '/' + savelist[0])
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
            params2save[key]=[]
            for idx, param in enumerate(params[key]):
                params2save[key][idx] = param.get_value()
        #file=open(path+'finall_model.npz','wb')
        np.savez(file,**params2save)
        #file.close()
config=ex({})

