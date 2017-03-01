# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import sys
sys.path.append('..')
import NNBuilder
from NNBuilder.DataPrepares import Load_mnist,Load_add
from NNBuilder.Algrithms import SGD
from NNBuilder.Extensions import Earlystop, Monitor ,Sample,Samples
from NNBuilder.Models import SoftmaxRegression
from NNBuilder.Model import Get_Model_Stream
from NNBuilder.MainLoop import Train
from NNBuilder.Visions.Visualization import get_result

a=Load_add()

