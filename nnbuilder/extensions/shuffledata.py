# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import copy
import numpy as np
from basic import ExtensionBase


class ShuffleData(ExtensionBase):
    def __init__(self):
        ExtensionBase.__init__(self)
        self.bucket = None
        self.scale = None

    def init(self):
        self.loaded = False
        if self.bucket is None:
            if self.scale is None:
                self.bucket = (self.data.size - 1) // self.config.batch_size + 1
            else:
                self.bucket = self.config.batch_size * self.scale

    def before_epoch(self):
        if self.loaded == False:
            minibatch_list = self.data.get_batch_indices(self.data.size, self.config.batch_size)
            minibatch_list = self.shuffle(minibatch_list)
            self.train_history['minibatch_list'] = minibatch_list
            self.logger("Shuffled Data At Epoch {} With Bucket {}".format(self.train_history['n_epoch'] + 1,
                                                                          self.bucket), 1, 1)
        else:
            self.loaded = False

    def shuffle(self, minibatch_list):
        shuffled_minibatch_list = []
        for i in range((len(minibatch_list) - 1) // self.bucket):
            minibatch_piece = copy.deepcopy(minibatch_list[i * self.bucket:(i + 1) * self.bucket])
            np.random.shuffle(minibatch_piece)
            shuffled_minibatch_list.extend(minibatch_piece)
        minibatch_piece = copy.deepcopy(minibatch_list[(len(minibatch_list) - 1) // self.bucket * self.bucket:])
        np.random.shuffle(minibatch_piece)
        shuffled_minibatch_list.extend(minibatch_piece)
        return shuffled_minibatch_list

    def load_(self, dict):
        self.loaded = True


shuffledata = ShuffleData()
