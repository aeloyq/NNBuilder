# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:02:46 2016

@author: aeloyq
"""

import pickle
import gzip
import theano
import numpy as np

''' define a function of xor '''

def xor_func(nums):
    num1=nums[0]%2
    num2=nums[1]%2
    if (num1==num2):
        return 0
    else:
        return 1

''' load mnist '''

def Load_mnist(configuration):
    with gzip.open(configuration['mnist_path'], 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
        return convert_to_theano_variable(configuration,train_set, valid_set, test_set)

''' load xor '''
            
def Load_xor(configuration):
    trainsets=[[],[]]
    for each_set in range(configuration['n_data']):
        each_input=list(configuration['rng'].randint(0,2,[configuration['n_inputs']]))
        each_output=xor_func(each_input)
        trainsets[0].append(each_input)
        trainsets[1].append(each_output)
    validsets=[[[0,0],[1,0],[0,1],[1,1]],[0,1,1,0]]
    testsets=[[[0,0],[1,0],[0,1],[1,1]],[0,1,1,0]]
    return convert_to_theano_variable(configuration,trainsets, validsets, testsets)

''' load theano variable '''
            
def convert_to_theano_variable(configuration,trainsets,validsets,testsets):
    train_X=theano.shared(np.array(trainsets[0],dtype="float32"))
    train_Y=theano.shared(np.array(trainsets[1],dtype="int32"))
    valid_X=theano.shared(np.array(validsets[0],dtype="float32"))
    valid_Y=theano.shared(np.array(validsets[1],dtype="int32"))
    test_X=theano.shared(np.array(testsets[0],dtype="float32"))
    test_Y=theano.shared(np.array(testsets[1],dtype="int32"))
    return [train_X, valid_X, test_X, train_Y, valid_Y, test_Y]