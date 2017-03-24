# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import numpy as np
import theano
import theano.tensor as t

a=t.matrix('a')
b=t.matrix('b')


o,u=theano.scan()
