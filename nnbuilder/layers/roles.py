# -*- coding: utf-8 -*-
"""
Created on  四月 26 21:15 2017

@author: aeloyq
"""


class Batch:
    def __str__(self):
        return 'batch'


batch = Batch()


class Time:
    def __str__(self):
        return 'time'


time = Time()


class Feature:
    def __str__(self):
        return 'feature'


feature = Feature()

class Channel:
    def __str__(self):
        return 'channel'


channel = Channel()

class Height:
    def __str__(self):
        return 'height'


height = Height()

class Width:
    def __str__(self):
        return 'width'


width = Width()


class Unit:
    def __str__(self):
        return 'unit'


unit = Unit()


class Search:
    def __str__(self):
        return 'search'


search = Search()


class Weight:
    attr = [None, unit]

    def __str__(self):
        return 'weight'


weight = Weight()

class NormWeight:
    attr = [unit]

    def __str__(self):
        return 'normweight'


normweight = NormWeight()


class Bias:
    attr = [unit]

    def __str__(self):
        return 'bias'


bias = Bias()


class Trans:
    attr = [unit, unit]

    def __str__(self):
        return 'trans'


trans = Trans()


class Convw:
    attr = [channel, channel, unit, unit]

    def __str__(self):
        return 'convw'


convw = Convw()


class Samplew:
    attr = [unit]

    def __str__(self):
        return 'samplew'


samplew = Samplew()


class Scan:
    def __str__(self):
        return 'scan'


scan = Scan()


class Onestep:
    def __str__(self):
        return 'Onestep'


onestep = Onestep()
