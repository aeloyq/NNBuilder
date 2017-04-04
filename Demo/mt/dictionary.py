# -*- coding: utf-8 -*-
"""
Created on  四月 04 22:59 2017

@author: aeloyq
"""
import numpy as np
from itertools import izip


def invert_dict3(d):
    return dict(izip(d.itervalues(), d.iterkeys()))

s='en'
t='fr'
ds=np.load('./data/vocab.{}-{}.{}.pkl'.format(s,t,s))
dt=np.load('./data/vocab.{}-{}.{}.pkl'.format(s,t,t))
dsv=invert_dict3(ds)
dst=invert_dict3(dt)
