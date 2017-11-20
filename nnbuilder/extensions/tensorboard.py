# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import timeit
import copy
import numpy as np
import nnbuilder
import os
import thread
from basic import *
from saveload import *
from earlystop import *
from collections import OrderedDict
from nnbuilder.tools import plotter


class TensorBoard(ExtensionBase):
    def __init__(self):
        ExtensionBase.__init__(self)
        self.freq = None
        self.compare_repo = None

    def init(self):
        self.path = './{}/plot/'.format(nnbuilder.config.name)
        self.compare_saving = []

        def load_data(file):
            return np.load(file)['mainloop'].tolist()

        if self.compare_repo is not None and self.compare_repo != []:
            if isinstance(self.compare_repo, (tuple, list)):
                for repo in self.compare_repo:
                    assert isinstance(repo, str)
                    savefile = './{}/save/final/final.npz'.format(repo)
                    if os.path.isfile(savefile):
                        self.compare_saving.append(load_data(savefile))
                    else:
                        path = './{}/save'.format(repo)
                        savelist = [path + name for name in os.listdir(path) if name.endswith('.npz')]
                        assert savelist > 0
                        savelist.sort(SaveLoad.compare_timestamp)
                        self.compare_saving.append(load_data(savelist[-1]))
            else:
                savelist = [self.compare_repo + name for name in os.listdir(self.compare_repo) if name.endswith('.npz')]
                for filename in savelist:
                    self.compare_saving.append(load_data(self.compare_repo + '/' + filename))

    def plot_model(self):
        kernel.printing(self.model.running_output, self.path + 'model/' + 'model.html')

    def plot_progress(self):
        current_saving = copy.deepcopy(self.train_history)
        if earlystop in self.extensions:
            current_saving['extensions'] = OrderedDict()
            current_saving['extensions']['EarlyStop'] = copy.deepcopy(earlystop.save_({}))
        thread.start_new_thread(plotter.monitor_progress, (current_saving, self.compare_saving,
                                                           self.path + 'progress/progress.html'))

    def before_train(self):
        if not os.path.exists('./{}/plot/model'.format(nnbuilder.config.name)): os.mkdir(
            './{}/plot/model'.format(nnbuilder.config.name))
        self.plot_model()

    def after_iteration(self):
        if self.freq is None:
            return
        else:
            if self.freq > 0:
                if self.train_history['n_iter'] % self.freq == 0:
                    self.plot_progress()

    def after_epoch(self):
        if self.freq is None:
            self.plot_progress()
        else:
            if self.freq < 0:
                if self.train_history['n_epoch'] % (-self.freq) == 0:
                    self.plot_progress()
            else:
                self.plot_progress()

    def after_train(self):
        self.plot_progress()


tensorboard = TensorBoard()
