# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
from extension import extension
from nnbuilder.main import mainloop


class ex(extension):
    def __init__(self, kwargs):
        extension.__init__(self, kwargs)
        self.shuffle_window = None

    def before_epoch(self):
        if self.kwargs['iter'] == 0:
            self.kwargs['minibatches'] = mainloop.get_minibatches(self.kwargs['datas'], True, self.shuffle_window)

    def after_epoch(self):
        self.kwargs['minibatches'] = mainloop.get_minibatches(self.kwargs['datas'], True,
                                                            self.shuffle_window)


config = ex({})
