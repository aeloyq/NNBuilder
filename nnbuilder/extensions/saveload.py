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
from basic import base


class ex(base):
    def __init__(self, kwargs):
        super(ex, self).__init__(kwargs)
        self.save_freq = 10000
        self.save_len = 3
        self.load = True
        self.save = True
        self.overwrite = True
        self.load_file_name = None
        self.save_epoch = False

    def init(self):
        super(ex, self).init()
        self.path = self.kwargs['config'].name + '/' + 'save' + '/'

    def before_train(self):
        kwargs = self.kwargs
        model = kwargs['model']
        if self.load:
            self.load_npz(model, mainloop_call=True)

    def after_iteration(self):
        if self.kwargs['n_iter'] % self.save_freq == 0:
            savename = '{}.npz'.format(self.kwargs['n_iter'])
            self.save_npz(self.kwargs['model'], '', savename, self.kwargs, self.save_len, self.overwrite, mainloop_call=True)

    def after_epoch(self):
        if self.save_epoch:
            savename = '{}.npz'.format(self.kwargs['n_epoch'])
            self.save_npz(self.kwargs['model'], 'epoch/', savename, self.kwargs, delete=False, mainloop_call=True)

    def after_train(self):
        self.save_npz(self.kwargs['model'], 'final/', 'final.npz', self.kwargs, self.overwrite, mainloop_call=True)

    def load_npz(self, model, lpath=None, name=None, mainloop_call=False):
        if mainloop_call:
            self.logger('Loading saved model from checkpoint:', 1, 1)
        else:
            print('Loading saved model from checkpoint:')
        # prepare loading
        layers = model.layers
        if lpath == None:
            path = self.path
        else:
            path = self.path + lpath
        if name == None:
            savelist = [path + name for name in os.listdir(path) if name.endswith('.npz')]
            if savelist == []:
                if mainloop_call:
                    self.logger('Checkpoint not found, exit loading', 2)
                else:
                    print('Checkpoint not found, exit loading')
                return
            savelist.sort(ex.compare_savefiles)
            lfile = savelist[-1]
        else:
            lfile = name
        if mainloop_call:
            self.logger('Checkpoint found : [{}]'.format(lfile), 2)
        else:
            print('Checkpoint found : [{}]'.format(lfile))
        # load params
        saves = np.load(lfile)
        sparams = saves['params'].tolist()
        soparams = saves['oparams'].tolist()
        for layername in layers:
            for param, sparam in zip(layers[layername].params, sparams[layername]):
                layers[layername].params[param].set(sparams[layername][sparam])
            for param, sparam in zip(layers[layername].oparams, soparams[layername]):
                layers[layername].oparams[param].set(soparams[layername][sparam])
        if mainloop_call:
            self.load_mainloop(saves['mainloop'].tolist(), saves['extensions'].tolist(), saves['optimizers'].tolist())

    def load_mainloop(self, smain, sextention, soptimizer):
        kwargs = self.kwargs
        kwargs['n_epoch'] = smain['n_epoch']
        kwargs['n_iter'] = smain['n_iter']
        kwargs['n_part'] = smain['n_part']
        kwargs['iter'] = smain['iter'] + 1
        kwargs['minibatches'] = smain['minibatches']
        kwargs['errors'] = smain['errors']
        kwargs['losses'] = smain['losses']
        kwargs['train_losses'] = smain['train_losses']
        for ex in kwargs['extensions']:
            extention_name = ex.__module__
            if extention_name in sextention:
                ex.config.load_(sextention[extention_name])
            else:
                self.logger('{} Load failed'.format(extention_name), 2)
        optimizer_name = kwargs['optimizer'].__class__.__name__
        if optimizer_name in soptimizer:
            kwargs['optimizer'].load_(soptimizer[optimizer_name])
        self.logger('Load sucessfully', 2)
        self.logger('', 2)

    def save_npz(self, model, spath, name, kwargs, savelen=1, delete=True, mainloop_call=False):
        # prepare save dictionary
        path = kwargs['config'].name + '/' + 'save' + '/' + spath
        layers = model.layers
        saves = OrderedDict()
        # save params
        trainable_params2save = OrderedDict()
        untrainable_params2save = OrderedDict()
        for layername in layers:
            trainable_params2save[layername] = OrderedDict()
            for pname, param in layers[layername].trainable_params.items():
                trainable_params2save[layername][pname] = param.get()
            untrainable_params2save[layername] = OrderedDict()
            for pname, param in layers[layername].untrainable_params.items():
                untrainable_params2save[layername][pname] = param.get()
        saves['trainable_params'] = trainable_params2save
        saves['untrainable_params'] = untrainable_params2save
        saves['mainloop'] = OrderedDict()
        saves['mainloop']['n_epoch'] = kwargs['n_epoch']
        saves['mainloop']['n_iter'] = kwargs['n_iter']
        saves['mainloop']['n_part'] = kwargs['n_part']
        saves['mainloop']['errors'] = kwargs['errors']
        saves['mainloop']['losses'] = kwargs['losses']
        saves['mainloop']['iter'] = kwargs['iter']
        saves['mainloop']['minibatches'] = kwargs['minibatches']
        saves['extentions'] = OrderedDict()
        # save extentions
        for ext in kwargs['extensions']:
            extention_name = ext.__name__.split('.')[-1]
            saves['extentions'][extention_name] = ext.config.save_(OrderedDict())
        # save optimizer
        saves['optimizers'] = OrderedDict()
        optimizer_name = kwargs['optimizer'].__class__.__name__
        saves['optimizers'][optimizer_name] = kwargs['optimizer'].save_(OrderedDict())
        # saving
        savename = path + name
        np.savez(savename, **saves)
        if mainloop_call:
            self.logger("")
            self.logger("Save Sucessfully At File : [{}]".format(savename), 1)
        # delete old files
        if delete:
            savelist = [path + name for name in os.listdir(path) if name.endswith('.npz')]
            savelist.sort(ex.compare_savefiles)
            for i in range(len(savelist) - savelen):
                savefile2delete = savelist[i]
                os.remove(savefile2delete)
                if mainloop_call:
                    self.logger("Deleted Old File : [{}]".format(savefile2delete), 1)
        if mainloop_call:
            self.logger("")

    @staticmethod
    def compare_savefiles(x, y):
        xt = os.stat(x)
        yt = os.stat(y)
        if xt.st_mtime > yt.st_mtime:
            return 1
        else:
            return -1


config = ex({})
instance = config
