# -*- coding: utf-8 -*-
"""
Created on  十一月 04 18:20 2017

@author: aeloyq
"""
from __future__ import absolute_import
from .. import *


def add_drop(model, drop, index):
    if drop is not None:
        if isinstance(drop, (tuple, list)):
            if drop[index] is not None:
                model.add(Dropout(drop[index]))
        else:
            model.add(Dropout(drop))


def add_norm(model, norm, index):
    if norm is not None:
        if isinstance(norm, (tuple, list)):
            if norm[index] is not None:
                model.add(BatchNorm(norm[index]))
        else:
            model.add(BatchNorm(norm))


def add_wt_decay(model, wt_decay, index):
    if wt_decay is not None:
        if isinstance(wt_decay, (tuple, list)):
            if wt_decay[index] is not None:
                model.add(layers.dropout(wt_decay[index]))
        else:
            model.add(layers.dropout(wt_decay))


def MLPcore(Mlp, hidden_size=(), activation=T.tanh,
            drop=None, norm=None, wt_decay=None, prefix=''):
    for i, d in enumerate(hidden_size):
        if isinstance(activation, (tuple, list)):
            Mlp.add(Hidden(d, activation=activation[i]), prefix + 'hidden_%d' % (i))
        else:
            Mlp.add(Hidden(d, activation=activation), prefix + 'hidden_%d' % (i))
        add_drop(Mlp, drop, i)
        add_norm(Mlp, norm, i)
        add_wt_decay(Mlp, wt_decay, i)
    return Mlp


def BinaryClassifier(PreModel, drop, norm, wt_decay, index=-1):
    PreModel.add(Logistic(1), 'output')
    add_drop(PreModel, drop, index)
    add_norm(PreModel, norm, index)
    add_wt_decay(PreModel, wt_decay, index)
    return PreModel


def Classifier(PreModel, output_size, drop, norm, wt_decay, index=-1):
    PreModel.add(Softmax(output_size), 'output')
    add_drop(PreModel, drop, index)
    add_norm(PreModel, norm, index)
    add_wt_decay(PreModel, wt_decay, index)
    return PreModel


def MultiLabelClassifier(PreModel, output_size, drop, norm, wt_decay, index=-1):
    PreModel.add(Logistic(output_size), 'output')
    add_drop(PreModel, drop, index)
    add_norm(PreModel, norm, index)
    add_wt_decay(PreModel, wt_decay, index)
    return PreModel


def Regressor(PreModel, output_size, drop, norm, wt_decay, index=-1):
    PreModel.add(Hidden(output_size, activation=None), 'output')
    add_drop(PreModel, drop, index)
    add_norm(PreModel, norm, index)
    add_wt_decay(PreModel, wt_decay, index)
    return PreModel


def MLPClassifier(input_size, output_size, hidden_size=(), activation=T.tanh, metrics=(metrics.accuracy,),
                  drop=None, norm=None, reg=None, wt_decay=None):
    Mlp = Model(X=X.num(input_size), Y=Y.cat(), lossfunction=lossfunctions.categorical_crossentropy, metrics=metrics)
    Mlp = MLPcore(Mlp, hidden_size=hidden_size, activation=activation,
                  drop=drop, norm=norm, wt_decay=wt_decay)
    Mlp = Classifier(Mlp, output_size, drop, norm, wt_decay)
    if reg is not None:
        Mlp.add(layers.regularization(reg))
    return Mlp


def MLPBinaryClassifier(input_size, hidden_size=(), activation=T.tanh, metrics=(metrics.accuracy,),
                        drop=None, norm=None, reg=None, wt_decay=None):
    Mlp = Model(X=X.num(input_size), Y=Y.cat(), lossfunction=lossfunctions.binary_crossentropy, metrics=metrics)
    Mlp = MLPcore(Mlp, hidden_size=hidden_size, activation=activation,
                  drop=drop, norm=norm, wt_decay=wt_decay)
    Mlp = BinaryClassifier(Mlp, drop, norm, wt_decay)
    if reg is not None:
        Mlp.add(layers.regularization(reg))
    return Mlp


def MLPMultiLabelClassifier(input_size, output_size, hidden_size=(), activation=T.tanh, metrics=(metrics.accuracy,),
                            drop=None, norm=None, reg=None, wt_decay=None):
    Mlp = Model(X=X.num(input_size), Y=Y.mulcat(output_size) , lossfunction=lossfunctions.binary_crossentropy, metrics=metrics)
    Mlp = MLPcore(Mlp, hidden_size=hidden_size, activation=activation,
                  drop=drop, norm=norm, wt_decay=wt_decay)
    Mlp = MultiLabelClassifier(Mlp, output_size, drop, norm, wt_decay)
    if reg is not None:
        Mlp.add(layers.regularization(reg))
    return Mlp


def MLPRegressor(input_size, output_size, hidden_size=(), activation=T.tanh, metrics=(metrics.accuracy,),
                 drop=None, norm=None, reg=None, wt_decay=None):
    Mlp = Model(X=X.num(input_size), Y=Y.num(), lossfunction=lossfunctions.mean_square_error, metrics=metrics)
    Mlp = MLPcore(Mlp, hidden_size=hidden_size, activation=activation,
                  drop=drop, norm=norm, wt_decay=wt_decay)
    Mlp.add(Softmax(output_size), 'output')
    Mlp = Regressor(Mlp, output_size, drop, norm, wt_decay)
    if reg is not None:
        Mlp.add(layers.regularization(reg))
    return Mlp
