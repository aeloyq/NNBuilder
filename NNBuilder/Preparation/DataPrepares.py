# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:02:46 2016

@author: aeloyq
"""

import pickle
import gzip
import theano
import theano.tensor as T
import numpy as np
import Recurrent_Dataprepare
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
        return [[train_set[0]], valid_set[0], test_set[0], train_set[1], valid_set[1], test_set[1]]
        #return convert_to_theano_variable(configuration,train_set, valid_set, test_set)

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


''' load add fun data '''

def Load_add(configuration):
    trainsets,validsets,testsets = [[], []],[[], []],[[], []]
    for each_set in range(10000):
        each_input = list(configuration['rng'].randint(0, 512, [2]))
        each_output = each_input[0]+each_input[1]
        trainsets[0].append(each_input)
        trainsets[1].append(each_output)
    for each_set in range(100):
        each_input = list(configuration['rng'].randint(0, 512, [2]))
        each_output = each_input[0]+each_input[1]
        validsets[0].append(each_input)
        validsets[1].append(each_output)
    for each_set in range(100):
        each_input = list(configuration['rng'].randint(0, 512, [2]))
        each_output = each_input[0]+each_input[1]
        testsets[0].append(each_input)
        testsets[1].append(each_output)
    return [[trainsets[0]], validsets[0], testsets[0], trainsets[1], validsets[1], testsets[1]]

''' load theano variable '''
            
def convert_to_theano_variable(configuration,trainsets,validsets,testsets):
    train_X=theano.shared(np.asarray(trainsets[0],dtype=theano.config.floatX),borrow=True)
    train_Y=theano.shared(np.asarray(trainsets[1],dtype=theano.config.floatX),borrow=True)
    valid_X=theano.shared(np.asarray(validsets[0],dtype=theano.config.floatX),borrow=True)
    valid_Y=theano.shared(np.asarray(validsets[1],dtype=theano.config.floatX),borrow=True)
    test_X=theano.shared(np.asarray(testsets[0],dtype=theano.config.floatX),borrow=True)
    test_Y=theano.shared(np.asarray(testsets[1],dtype=theano.config.floatX),borrow=True)
    return [[train_X], valid_X, test_X, T.cast(train_Y, 'int32'), T.cast(valid_Y, 'int32'),T.cast(test_Y, 'int32') ]