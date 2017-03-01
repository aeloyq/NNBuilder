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
from Preparation import *
import config

''' define a function of xor '''

def xor_func(nums):
    num1=nums[0]%2
    num2=nums[1]%2
    if (num1==num2):
        return 0
    else:
        return 1

''' load mnist '''

def Load_mnist():
    with gzip.open(config.data_path, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
        return [train_set[0], valid_set[0], test_set[0], train_set[1], valid_set[1], test_set[1]]

''' load xor '''
            
def Load_xor():
    trainsets=[[],[]]
    for each_set in range(100):
        each_input=list(config.rng.randint(0,2,2))
        each_output=xor_func(each_input)
        trainsets[0].append(each_input)
        trainsets[1].append(each_output)
    validsets=[[[0,0],[1,0],[0,1],[1,1]],[0,1,1,0]]
    testsets=[[[0,0],[1,0],[0,1],[1,1]],[0,1,1,0]]
    return convert_to_theano_variable(trainsets, validsets, testsets)


''' load add fun data '''

def Load_add():
    maxlen=2
    trainsets,validsets,testsets = [[], []],[[], []],[[], []]
    def binary(n):
        bn = bin(n)[2:]
        b = np.zeros(10)
        for idx in range(len(bn)):
            b[-idx - 1] = bn[-idx - 1]
        return list(b)
    for each_set in range(10000):
        each_input = config.rng.randint(0, 512, [2])
        each_output = each_input[0]+each_input[1]
        inp=[binary(each_input[0]),binary(each_input[1])]
        each_output=binary(each_output)
        trainsets[0].append(inp)
        trainsets[1].append(each_output)
    for each_set in range(100):
        each_input = config.rng.randint(0, 512, [2])
        each_output = each_input[0]+each_input[1]
        inp = [binary(each_input[0]), binary(each_input[1])]
        each_output = binary(each_output)
        validsets[0].append(inp)
        validsets[1].append(each_output)
    for each_set in range(100):
        each_input = config.rng.randint(0, 512, [2])
        each_output = each_input[0]+each_input[1]
        inp = [binary(each_input[0]), binary(each_input[1])]
        each_output = binary(each_output)
        testsets[0].append(inp)
        testsets[1].append(each_output)
    return [trainsets[0], validsets[0], testsets[0], trainsets[1], validsets[1], testsets[1]]

''' load add fun data '''

def Load_imdb(n_words=100000,valid_portion=0.1,maxlen=None,sort_by_len=True):
    f = open(config.data_path, 'rb')
    train_set = pickle.load(f)
    test_set = pickle.load(f)
    f.close()
    if maxlen:
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set[0], train_set[1]):
            if len(x) < maxlen:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = (new_train_set_x, new_train_set_y)
        del new_train_set_x, new_train_set_y

    # split training set into validation set
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.random.permutation(n_samples)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]
    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)

    def remove_unk(x):
        return [[1 if w >= n_words else w for w in sen] for sen in x]

    test_set_x, test_set_y = test_set
    valid_set_x, valid_set_y = valid_set
    train_set_x, train_set_y = train_set

    train_set_x = remove_unk(train_set_x)
    valid_set_x = remove_unk(valid_set_x)
    test_set_x = remove_unk(test_set_x)

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        sorted_index = len_argsort(test_set_x)
        test_set_x = [test_set_x[i] for i in sorted_index]
        test_set_y = [test_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set_x)
        valid_set_x = [valid_set_x[i] for i in sorted_index]
        valid_set_y = [valid_set_y[i] for i in sorted_index]

        sorted_index = len_argsort(train_set_x)
        train_set_x = [train_set_x[i] for i in sorted_index]
        train_set_y = [train_set_y[i] for i in sorted_index]
    return [train_set_x,valid_set_x, test_set_x, train_set_y, valid_set_y, test_set_y]

''' load theano variable '''
            
def convert_to_theano_variable(trainsets,validsets,testsets):
    train_X=theano.shared(np.asarray(trainsets[0],dtype=theano.config.floatX),borrow=True)
    train_Y=theano.shared(np.asarray(trainsets[1],dtype=theano.config.floatX),borrow=True)
    valid_X=theano.shared(np.asarray(validsets[0],dtype=theano.config.floatX),borrow=True)
    valid_Y=theano.shared(np.asarray(validsets[1],dtype=theano.config.floatX),borrow=True)
    test_X=theano.shared(np.asarray(testsets[0],dtype=theano.config.floatX),borrow=True)
    test_Y=theano.shared(np.asarray(testsets[1],dtype=theano.config.floatX),borrow=True)
    return [train_X, valid_X, test_X, T.cast(train_Y, 'int32'), T.cast(valid_Y, 'int32'),T.cast(test_Y, 'int32') ]