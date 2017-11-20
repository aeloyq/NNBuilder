# -*- coding: utf-8 -*-
"""
Created on Sat Dec 24 13:45:34 2016

@author: aeloyq
"""
import os
from basic import *


class SaveLoad(ExtensionBase):
    def __init__(self):
        ExtensionBase.__init__(self)
        self.max = 3
        self.freq = 10000
        self.save = True
        self.load = True
        self.epoch = False
        self.overwrite = True
        self.loadfile = None

    def init(self):
        self.path = './'+self.config.name+'/save/'

    def before_train(self):
        if self.load:
            self.mainloop_load(self.model, '')

    def after_iteration(self):
        if (self.train_history['n_iter']) % self.freq == 0:
            savename = '{}.npz'.format(self.train_history['n_iter'])
            self.mainloop_save(self.model, '', savename, self.max, self.overwrite)

    def after_epoch(self):
        if self.epoch:
            savename = '{}.npz'.format(self.train_history['n_epoch'])
            self.mainloop_save(self.model, 'epoch/', savename, self.max, self.overwrite)

    def after_train(self):
        self.mainloop_save(self.model, 'final/', 'final.npz', self.max, self.overwrite)

    def mainloop_save(self, model, path, file, max=1, overwrite=True):
        filepath = self.path + path + file
        np.savez(filepath,
                 parameter=SaveLoad.get_params(model),
                 train_history=SaveLoad.get_train_history(self.train_history),
                 extensions=SaveLoad.get_extensions_dict(self.extensions),
                 optimizer=SaveLoad.get_optimizer_dict(self.model.optimizer))
        if self.is_log_detail():
            self.logger("")
        self.logger("Save Sucessfully At File : [{}]".format(filepath), 1)
        # delete old files
        if overwrite:
            filelist = [self.path + path + name for name in os.listdir(self.path + path) if name.endswith('.npz')]
            filelist.sort(SaveLoad.compare_timestamp)
            for i in range(len(filelist) - max):
                os.remove(filelist[i])
                if self.is_log_detail():
                    self.logger("Deleted Old File : [{}]".format(filelist[i]), 1)
        if self.is_log_detail():
            self.logger("")

    def mainloop_load(self, model, file):
        self.logger('Loading saved model from checkpoint:', 1, 1)
        # prepare loading
        if os.path.isfile(file):
            file = self.path + file
        else:
            filelist = [self.path + filename for filename in os.listdir(self.path + file) if filename.endswith('.npz')]
            if filelist == []:
                self.logger('Checkpoint not found, exit loading', 2)
                return
            filelist.sort(SaveLoad.compare_timestamp)
            file = filelist[-1]

        self.logger('Checkpoint found : [{}]'.format(file), 2)
        # load params
        SaveLoad.load_params(model, file)
        SaveLoad.load_train_history(self.train_history, file)
        SaveLoad.load_extensions(self.extensions, file)
        SaveLoad.load_optimizer(self.model.optimizer, file)
        self.logger('Load sucessfully', 2)
        self.logger('', 2)

    @staticmethod
    def get_params(model):
        params = OrderedDict()
        for name, param in model.params.items():
            params[name] = param().get()
        return params

    @staticmethod
    def get_train_history(train_history):
        return train_history

    @staticmethod
    def get_extensions_dict(extensions):
        extensions_dict = OrderedDict()
        for ex in extensions:
            extensions_dict[ex.__class__.__name__] = ex.save_(OrderedDict())
        return extensions_dict

    @staticmethod
    def get_optimizer_dict(optimizer):
        optimizer_dict = OrderedDict()
        optimizer_dict[optimizer.__class__.__name__] = optimizer.save_(OrderedDict())
        return optimizer_dict

    @staticmethod
    def load_params(model, file):
        params = np.load(file)['parameter'].tolist()
        for name, param in params.items():
            model.params[name]().set(param)

    @staticmethod
    def load_train_history(train_history, file):
        loaded_train_history = np.load(file)['train_history'].tolist()
        for key, value in loaded_train_history.items():
            train_history[key] = value

    @staticmethod
    def load_extensions(extensions, file):
        loaded_extensions = np.load(file)['extensions'].tolist()
        for ex in extensions:
            if ex.__class__.__name__ in loaded_extensions:
                ex.load_(loaded_extensions[ex.__class__.__name__])

    @staticmethod
    def load_optimizer(optimizer, file):
        loaded_optimizer = np.load(file)['optimizer'].tolist()
        if optimizer.__class__.__name__ in loaded_optimizer:
            optimizer.load_(loaded_optimizer[optimizer.__class__.__name__])

    @staticmethod
    def save_file(model, file):
        params = SaveLoad.get_params(model)
        np.savez(file, parameter=params)

    @staticmethod
    def compare_timestamp(x, y):
        xt = os.stat(x)
        yt = os.stat(y)
        if xt.st_mtime > yt.st_mtime:
            return 1
        else:
            return -1


saveload = SaveLoad()
