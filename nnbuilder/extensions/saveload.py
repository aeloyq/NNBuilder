# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:45:34 2016

@author: aeloyq
"""
import numpy as np
import os
import nnbuilder
import timeit
from collections import OrderedDict
from extension import extension


class ex(extension):
    def __init__(self,kwargs):
        extension.__init__(self,kwargs)
        self.save_freq = 10000
        self.save_len = 3
        self.load = True
        self.save = True
        self.overwrite = True
        self.load_file_name = ''
        self.save_epoch = False
        self.data_changed=False

    def init(self):
        extension.init(self)

    def before_train(self):
        kwargs = self.kwargs
        model = kwargs['model']
        layers = model.layers
        path = './%s/save' % (nnbuilder.config.name)
        self.path = path
        if self.load:
            self.logger('Loading saved model from checkpoint:', 1, 1)
            if self.load_file_name == '':
                savelist = [name for name in os.listdir(path) if name.endswith('.npz')]
                if savelist == []:
                    self.logger('Checkpoint not found, exit loading', 2)
                    return

                def cp(x, y):
                    xt = os.stat(path + '/'+ x)
                    yt = os.stat(path + '/'+ y)
                    if xt.st_mtime > yt.st_mtime:
                        return 1
                    else:
                        return -1
                savelist.sort(cp)
                self.load_file_name = savelist[-1]
            self.logger('Checkpoint is found:{}...'.format(self.load_file_name), 2)
            params = (np.load(path + '/' + self.load_file_name))['save'].tolist()
            for key in layers:
                for param, sparam in zip(layers[key].params, params[key]):
                    layers[key].params[param].set_value(params[key][sparam])
            kwargs['n_epoch'] = params['n_epoch']
            if not self.data_changed:
                kwargs['n_iter'] = params['n_iter']
                kwargs['n_part'] = params['n_part']
                kwargs['iter'] = params['iter'] + 1
                kwargs['minibatches'] = params['minibatches']
            for i in params['errors']:
                kwargs['errors'].append(i)
            for i in params['costs']:
                kwargs['costs'].append(i)
            name = ''
            try:
                for ex in kwargs['extensions']:
                    ex.config.load_(params)
                kwargs['optimizer'].load_(params)
            except:
                self.logger('{} Load failed'.format(name), 2)
            self.logger('Load sucessfully', 2)
            self.logger('',2)

    def after_iteration(self):
        kwargs = self.kwargs
        if kwargs['n_iter'] % self.save_freq == 0:
            savename = self.path + '/%s.npz' % (kwargs['n_iter'])
            self.save_npz(savename, self.overwrite)

    def after_epoch(self):
        if self.save_epoch:
            savename = self.path + '/epoch/{}.npz'.format(self.kwargs['n_epoch'])
            self.save_npz(savename, self.overwrite)

    def after_train(self):
        self.save_npz(self.path + '/finall', self.overwrite)

    def load_npz(self, model, name=''):

        layers = model.layers
        path = './%s/save' % (nnbuilder.config.name)
        print('Loading saved model from checkpoint:')
        if name == '':
            savelist = [name for name in os.listdir(path) if name.endswith('.npz')]
            if savelist == []:
                print('Checkpoint not found, exit loading')
                return

            def cp(x, y):
                xt = os.stat(path + '/'+ x)
                yt = os.stat(path + '/'+ y)
                if xt.st_mtime > yt.st_mtime:
                    return 1
                else:
                    return -1

            savelist.sort(cp)
            self.load_file_name = savelist[-1]
            print('Checkpoint is found : [{}]'.format(self.load_file_name))
        else:
            self.load_file_name = name + '.npz'
            print('Checkpoint is found : [{}]'.format(self.load_file_name))
        params = (np.load(path + '/' + self.load_file_name))['save'].tolist()
        for key in layers:
            for param, sparam in zip(layers[key].params, params[key]):
                layers[key].params[param].set_value(params[key][sparam])

    def save_npz(self, name, delete=True):
        kwargs = self.kwargs
        self.logger("Prepare to save model", 1, 1)
        model = kwargs['model']
        layers = model.layers
        params = OrderedDict()
        params2save = OrderedDict()
        for key in layers:
            params[key] = layers[key].params
        for key in params:
            params2save[key] = OrderedDict()
            for pname, param in params[key].items():
                params2save[key][pname] = param.get_value()
        params2save['n_epoch'] = kwargs['n_epoch']
        params2save['n_iter'] = kwargs['n_iter']
        params2save['n_part'] = kwargs['n_part']
        params2save['errors'] = kwargs['errors']
        params2save['costs'] = kwargs['costs']
        params2save['iter'] = kwargs['iter']
        params2save['minibatches'] = kwargs['minibatches']
        for ex in kwargs['extensions']:
            ex.config.save_(params2save)
        kwargs['optimizer'].save_(params2save)
        savename = name
        np.savez(savename, save=params2save)
        self.logger("Save Sucessfully At File : [{}]".format(savename), 2)
        if delete:
            savelist = [name for name in os.listdir(self.path) if name.endswith('.npz')]

            def cp(x, y):
                xt = os.stat(self.path + '/' + x)
                yt = os.stat(self.path + '/' + y)
                if xt.st_mtime > yt.st_mtime:
                    return 1
                else:
                    return -1

            savelist.sort(cp)
            for i in range(len(savelist) - self.save_len):
                os.remove(self.path + '/' + savelist[i])
                self.logger("Deleted Old File : [{}]".format(self.path + '/' + savelist[i]), 2)
        self.logger("",2)


config = ex({})
