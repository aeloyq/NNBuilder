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

def mt_sample(inputs,result,y):
    sample_str = "Sample:\r\n"
    sst=''
    tst=''
    mst=''
    for i in inputs:
        sst=sst+dsv[i[0]]+' '
    for i in y:
        tst=tst+dst[i[0]]+' '
    for i in result:
        mst=mst+dst[i[0]]+' '
        if i[0]==0:break
    sample_str += 'Source Sentence: {}\r\n'.format(sst)
    sample_str += 'Sample Sentence: {}'.format(mst)
    return sample_str,tst