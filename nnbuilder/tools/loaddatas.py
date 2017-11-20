# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 17:02:46 2016

@author: aeloyq
"""
import os
import gzip
import pickle
import numpy as np


def Load_stream(path, n, endwith='.npy'):
    vtdata = path + '/' + 'ValidTest' + endwith
    stream = []
    for i in range(n):
        stream.append(path + '/' + 'TrainData{}{}'.format(str(i + 1).zfill(3), endwith))
    return vtdata, stream


def load_data(data):
    train_x, valid_x, test_x, train_y, valid_y, test_y = data
    return [train_x, train_y], [valid_x, valid_y], [test_x, test_y]


def Load_npz(path, name=None):
    if name == None:
        savelist = [name for name in os.listdir(path) if name.endswith('.npz')]

        def cp(x, y):
            xt = os.stat(path + '/' + x)
            yt = os.stat(path + '/' + y)
            if xt.st_mtime > yt.st_mtime:
                return 1
            else:
                return -1

        savelist.sort(cp)
        load_file_name = savelist[-1]
    else:
        load_file_name = path + '/' + name
    return np.load(load_file_name)['save'].tolist()


''' define a function of xor '''


def xor_func(nums):
    num1 = nums[0] % 2
    num2 = nums[1] % 2
    if (num1 == num2):
        return 0
    else:
        return 1


''' load mnist '''


def transfer_image(imagesize, train_X, valid_X, test_X):
    n_train = len(train_X)
    n_valid = len(valid_X)
    n_test = len(test_X)
    train_X = np.reshape(train_X, [n_train] + imagesize)
    valid_X = np.reshape(valid_X, [n_valid] + imagesize)
    test_X = np.reshape(test_X, [n_test] + imagesize)
    return train_X, valid_X, test_X


def Download_mnist(data_path):
    if not os.path.isfile(data_path):
        from six.moves import urllib
        origin = (
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, data_path)


def Download_imdb(data_path):
    if not os.path.isfile(data_path):
        from six.moves import urllib
        origin = (
            "http://www.iro.umontreal.ca/~lisa/deep/data/imdb.pkl"
        )
        print('Downloading data from %s' % origin)
        urllib.request.urlretrieve(origin, data_path)


def Load_mnist(data_path):
    Download_mnist(data_path)
    with gzip.open(data_path, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)

        return load_data([train_set[0], valid_set[0], test_set[0], train_set[1], valid_set[1], test_set[1]])


def Load_mnist_image(data_path):
    Download_mnist(data_path)
    with gzip.open(data_path, 'rb') as f:
        try:
            train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        except:
            train_set, valid_set, test_set = pickle.load(f)
        train_X = train_set[0]
        train_Y = train_set[1]
        valid_X = valid_set[0]
        valid_Y = valid_set[1]
        test_X = test_set[0]
        test_Y = test_set[1]
        train_X, valid_X, test_X = transfer_image([1, 28, 28], train_X, valid_X, test_X)
        return load_data([train_X, valid_X, test_X, train_Y, valid_Y, test_Y])


''' load xor '''


def Load_xor():
    trainsets = [[], []]
    for each_set in range(100):
        each_input = list(np.random.randint(0, 2, 2))
        each_output = xor_func(each_input)
        trainsets[0].append(each_input)
        trainsets[1].append(each_output)
    validsets = [[[0, 0], [1, 0], [0, 1], [1, 1]], [0, 1, 1, 0]]
    testsets = [[[0, 0], [1, 0], [0, 1], [1, 1]], [0, 1, 1, 0]]
    return load_data([trainsets[0], validsets[0], testsets[0], trainsets[1], validsets[1], testsets[1]])


''' load add fun data '''


def Load_add():
    maxlen = 2
    trainsets, validsets, testsets = [[], []], [[], []], [[], []]

    def binary(n):
        bn = bin(n)[2:]
        b = np.zeros(10)
        for idx in range(len(bn)):
            b[-idx - 1] = bn[-idx - 1]
        return list(b)

    for each_set in range(100000):
        each_input = np.random.randint(0, 512, [2])
        each_output = each_input[0] + each_input[1]
        inp = [binary(each_input[0]), binary(each_input[1])]
        each_output = binary(each_output)
        trainsets[0].append(inp)
        trainsets[1].append(each_output)
    for each_set in range(10000):
        each_input = np.random.randint(0, 512, [2])
        each_output = each_input[0] + each_input[1]
        inp = [binary(each_input[0]), binary(each_input[1])]
        each_output = binary(each_output)
        validsets[0].append(inp)
        validsets[1].append(each_output)
    for each_set in range(10000):
        each_input = np.random.randint(0, 512, [2])
        each_output = each_input[0] + each_input[1]
        inp = [binary(each_input[0]), binary(each_input[1])]
        each_output = binary(each_output)
        testsets[0].append(inp)
        testsets[1].append(each_output)
    return load_data([trainsets[0], validsets[0], testsets[0], trainsets[1], validsets[1], testsets[1]])


''' load add fun data '''


def Load_imdb(data_path, n_words=10000, valid_portion=0.05, maxlen=None, sort_by_len=True, test_size=500):
    Download_imdb(data_path)
    f = open(data_path, 'rb')
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

    test = (test_set_x, test_set_y)
    if test_size > 0:
        # The test set is sorted by size, but we want to keep random
        # size example.  So we must select a random selection of the
        # examples.
        idx = np.arange(len(test[0]))
        np.random.shuffle(idx)
        idx = idx[:test_size]
        test = ([test[0][n] for n in idx], [test[1][n] for n in idx])
    test_set_x, test_set_y = test

    return load_data([train_set_x, valid_set_x, test_set_x, train_set_y, valid_set_y, test_set_y])


def Load_mt(data_path, maxlen=None, sort_by_len=True, sort_by_asc=True):
    print('\r\nloading data...')
    path = data_path
    data = np.load(path)
    train_set, valid_set, test_set, train_sety, valid_sety, test_sety = data['arr_0']
    if maxlen:
        print('clipping to max length...')
        new_train_set_x = []
        new_train_set_y = []
        for x, y in zip(train_set, train_sety):
            if len(x) < maxlen + 1 and len(y) < maxlen + 1:
                new_train_set_x.append(x)
                new_train_set_y.append(y)
        train_set = new_train_set_x
        train_sety = new_train_set_y
        del new_train_set_x, new_train_set_y
        print('clipped')

    def len_argsort(seq):
        return sorted(range(len(seq)), key=lambda x: len(seq[x]))

    if sort_by_len:
        print('sorting by length...')
        sorted_index = len_argsort(test_set)
        if not sort_by_asc: sorted_index.reverse()
        test_set = [test_set[i] for i in sorted_index]
        test_sety = [test_sety[i] for i in sorted_index]

        sorted_index = len_argsort(valid_set)
        if not sort_by_asc: sorted_index.reverse()
        valid_set = [valid_set[i] for i in sorted_index]
        valid_sety = [valid_sety[i] for i in sorted_index]

        sorted_index = len_argsort(train_set)
        if not sort_by_asc: sorted_index.reverse()
        train_set = [train_set[i] for i in sorted_index]
        train_sety = [train_sety[i] for i in sorted_index]
        print('sorted')
    print('Data loaded,ok')
    print('Train sentence pairs loaded in total: %s' % (len(train_set)))
    print('Valid sentence pairs loaded in total: %s' % (len(valid_set)))
    print('Test sentence pairs loaded in total: %s' % (len(test_set)))
    print('Max length in trainsets: %s' % max([len(i) for i in train_set]))
    print('Mean length in trainsets: %s' % np.mean([len(i) for i in train_set]))
    return load_data([train_set, valid_set, test_set, train_sety, valid_sety, test_sety])
