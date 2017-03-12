# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import sys
sys.path.append('..')
import nnbuilder
from nnbuilder.dataprepares import Load_mnist,Load_add
from nnbuilder.algrithms import sgd
from nnbuilder.extensions import earlystop, monitor ,sample,samples
from nnbuilder.models import softmaxregression
from nnbuilder.model import Get_Model_Stream
from nnbuilder.mainloop import train
from nnbuilder.visions.Visualization import get_result

a=Load_add()

