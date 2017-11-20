# -*- coding: utf-8 -*-
"""
Created on  九月 11 6:34 2017

@author: aeloyq
"""
from standard import *


def LeNetCore(Lenet, convdrop=None, ffdrop=None,
              norm=None, wt_decay=None, asym=False, prefix=''):
    Lenet.add(conv(nfilters=6, filtersize=[5, 5]), prefix + 'conv1')
    Lenet.add(subsample(windowsize=(2, 2)), prefix + 'pool1')
    if asym:
        Lenet.add(asymconv(filters=[(3, 3), (4, 2), (2, 1), (6, 0)], filtersize=[5, 5]), prefix + 'conv2')
    else:
        Lenet.add(conv(nfilters=16, filtersize=[5, 5]), prefix + 'conv2')
    if convdrop is not None:
        Lenet.add(layers.dropout(convdrop))
    Lenet.add(subsample(windowsize=(2, 2)), prefix + 'pool2')
    Lenet.add(conv(nfilters=120, filtersize=[4, 4]), prefix + 'conv3')
    Lenet.add(flatten(), prefix + 'flatten')
    Lenet.add(hnn(84, activation=T.tanh), prefix + 'hidden')
    add_drop(Lenet, ffdrop, 0)
    add_norm(Lenet, norm, 0)
    add_wt_decay(Lenet, wt_decay, 0)
    return Lenet


def LeNetClassifier(input_size, output_size, convdrop=None, ffdrop=None,
                    metrics=(metrics.accuracy,), norm=None, wt_decay=None, asym=False):
    Lenet = Model(input_size, X=var.X.image, metrics=metrics)
    Lenet = LeNetCore(Lenet, convdrop=convdrop, ffdrop=ffdrop,
                      norm=norm, wt_decay=wt_decay, asym=asym)
    Lenet = Classifier(Lenet, output_size, ffdrop, norm, wt_decay)
    return Lenet


def LeNetMutiLabelClassifier(input_size, output_size, convdrop=None, ffdrop=None,
                             metrics=(metrics.accuracy,), norm=None, wt_decay=None, asym=False):
    Lenet = Model(input_size, X=var.X.image, Y=var.Y.encodedlabel,
                  lossfunction=lossfunctions.binary_crossentropy, metrics=metrics)
    Lenet = LeNetCore(Lenet, convdrop=convdrop, ffdrop=ffdrop,
                      norm=norm, wt_decay=wt_decay, asym=asym)
    Lenet = MultiLabelClassifier(Lenet, output_size, ffdrop, norm, wt_decay)
    return Lenet


def AlexNetCore(AlexNet, drop=0.5, prefix=''):
    AlexNet.add(conv(nfilters=96, filtersize=[11, 11], strides=(4, 4)), prefix + 'conv1')
    AlexNet.add(activation(activation=T.relu), prefix + 'relu1')
    AlexNet.add(pool(windowsize=(3, 3), strides=(2, 2)), prefix + 'pool1')
    AlexNet.add(layers.lrn())
    AlexNet.add(conv(nfilters=256, filtersize=[5, 5], strides=(4, 4), pad=2), prefix + 'conv2')
    AlexNet.add(activation(activation=T.relu), prefix + 'relu2')
    AlexNet.add(pool(windowsize=(3, 3), strides=(2, 2)), prefix + 'pool2')
    AlexNet.add(layers.lrn())
    AlexNet.add(conv(nfilters=384, filtersize=[3, 3], strides=(4, 4)), prefix + 'conv3')
    AlexNet.add(activation(activation=T.relu), prefix + 'relu3')
    AlexNet.add(conv(nfilters=384, filtersize=[3, 3], strides=(4, 4)), prefix + 'conv4')
    AlexNet.add(activation(activation=T.relu), prefix + 'relu4')
    AlexNet.add(conv(nfilters=256, filtersize=[3, 3], strides=(4, 4)), prefix + 'conv5')
    AlexNet.add(activation(activation=T.relu), prefix + 'relu5')
    AlexNet.add(pool(windowsize=(3, 3), strides=(2, 2)), prefix + 'pool5')
    AlexNet.add(flatten(), prefix + 'flatten6')
    AlexNet.add(hnn(4096, activation=None), prefix + 'hidden6')
    AlexNet.add(activation(activation=T.relu), prefix + 'relu6')
    add_drop(AlexNet, drop, 0)
    AlexNet.add(hnn(4096, activation=None), prefix + 'hidden7')
    AlexNet.add(activation(activation=T.relu), prefix + 'relu7')
    add_drop(AlexNet, drop, 1)
    return AlexNet


def Vgg16Core(Lenet, convdrop=None, ffdrop=None,
              norm=None, wt_decay=None, asym=False, prefix=''):
    Lenet.add(conv(nfilters=6, filtersize=[5, 5]), prefix + 'conv1')
    Lenet.add(subsample(windowsize=(2, 2)), prefix + 'pool1')
    if asym:
        Lenet.add(asymconv(filters=[(3, 3), (4, 2), (2, 1), (6, 0)], filtersize=[5, 5]), prefix + 'conv2')
    else:
        Lenet.add(conv(nfilters=16, filtersize=[5, 5]), prefix + 'conv2')
    if convdrop is not None:
        Lenet.add(layers.dropout(convdrop))
    Lenet.add(subsample(windowsize=(2, 2)), prefix + 'pool2')
    Lenet.add(conv(nfilters=120, filtersize=[4, 4]), prefix + 'conv3')
    Lenet.add(flatten(), prefix + 'flatten')
    Lenet.add(hnn(84, activation=T.tanh), prefix + 'hidden')
    add_drop(Lenet, ffdrop, 0)
    add_norm(Lenet, norm, 0)
    add_wt_decay(Lenet, wt_decay, 0)
    return Lenet


def Vgg19Core(Lenet, convdrop=None, ffdrop=None,
              norm=None, wt_decay=None, asym=False, prefix=''):
    Lenet.add(conv(nfilters=6, filtersize=[5, 5]), prefix + 'conv1')
    Lenet.add(subsample(windowsize=(2, 2)), prefix + 'pool1')
    if asym:
        Lenet.add(asymconv(filters=[(3, 3), (4, 2), (2, 1), (6, 0)], filtersize=[5, 5]), prefix + 'conv2')
    else:
        Lenet.add(conv(nfilters=16, filtersize=[5, 5]), prefix + 'conv2')
    if convdrop is not None:
        Lenet.add(layers.dropout(convdrop))
    Lenet.add(subsample(windowsize=(2, 2)), prefix + 'pool2')
    Lenet.add(conv(nfilters=120, filtersize=[4, 4]), prefix + 'conv3')
    Lenet.add(flatten(), prefix + 'flatten')
    Lenet.add(hnn(84, activation=T.tanh), prefix + 'hidden')
    add_drop(Lenet, ffdrop, 0)
    add_norm(Lenet, norm, 0)
    add_wt_decay(Lenet, wt_decay, 0)
    return Lenet


def ResNetCore(Lenet, convdrop=None, ffdrop=None,
               norm=None, wt_decay=None, asym=False, prefix=''):
    Lenet.add(conv(nfilters=6, filtersize=[5, 5]), prefix + 'conv1')
    Lenet.add(subsample(windowsize=(2, 2)), prefix + 'pool1')
    if asym:
        Lenet.add(asymconv(filters=[(3, 3), (4, 2), (2, 1), (6, 0)], filtersize=[5, 5]), prefix + 'conv2')
    else:
        Lenet.add(conv(nfilters=16, filtersize=[5, 5]), prefix + 'conv2')
    if convdrop is not None:
        Lenet.add(layers.dropout(convdrop))
    Lenet.add(subsample(windowsize=(2, 2)), prefix + 'pool2')
    Lenet.add(conv(nfilters=120, filtersize=[4, 4]), prefix + 'conv3')
    Lenet.add(flatten(), prefix + 'flatten')
    Lenet.add(hnn(84, activation=T.tanh), prefix + 'hidden')
    add_drop(Lenet, ffdrop, 0)
    add_norm(Lenet, norm, 0)
    add_wt_decay(Lenet, wt_decay, 0)
    return Lenet
