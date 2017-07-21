# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 23:24:31 2016

@author: aeloyq
"""
import numpy as np
import nnbuilder
from nnbuilder.layers.roles import weight,bias



def plot(self, costs, errors, params, roles):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pylab as plt
    x_axis = np.arange(len(costs)) + 1

    plt.figure(1)
    plt.cla()
    plt.title(nnbuilder.config.name)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.plot(x_axis, costs, label='Loss', color='orange')
    plt.legend()
    plt.savefig(self.path + 'process_cost.png')

    plt.figure(2)
    plt.cla()
    plt.title(nnbuilder.config.name)
    plt.ylabel('Error')
    plt.xlabel('Epochs')
    plt.plot(x_axis, errors, label='Error', color='blue')
    plt.legend()
    plt.savefig(self.path + 'process_error.png')

    n_im = len(params)
    a = np.int(np.sqrt(n_im))
    b = a
    if a * b < n_im: a += 1
    if a * b < n_im: b += 1
    plt.figure(3, (b * 4, a * 4))
    plt.cla()

    i = 0
    for key, param in params.items():
        i += 1
        if roles[key] is weight:
            plt.subplot(a, b, i)
            value = param
            plt.title(key + ' ' + str(value.shape))
            img = np.asarray(value)
            if img.ndim != 1:
                plt.imshow(img, cmap='gray')
        elif roles[key] is bias:
            plt.subplot(a, b, i)
            y = param
            plt.title(key + ' ' + str(y.shape))
            x_axis_bi = np.arange(y.shape[0])
            plt.plot(x_axis_bi, y, color='black')
    plt.savefig(self.path + 'paramsplot.png')

    plt.cla()

def analysis_save():
    pass