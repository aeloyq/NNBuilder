# -*- coding: utf-8 -*-
"""
Created on  三月 17 20:20 2017

@author: aeloyq
"""
import nnbuilder.config as config
import nnbuilder.logger as logger
import cPickle as cp


def get_data_stream():
    source = config.source
    target = config.target
    source_sentence = './data/{}-{}.{}.tok'.format(source, target, source)
    target_sentence = './data/{}-{}.{}.tok'.format(source, target, target)
    source_vocab = './data/vocab.{}-{}.{}.pkl'.format(source, target, source)
    target_vocab = './data/vocab.{}-{}.{}.pkl'.format(source, target, target)
    source_vocab = cp.load(open(source_vocab, 'rb'))
    target_vocab = cp.load(open(target_vocab, 'rb'))
    source_ = open(source_sentence, 'rb')
    target_ = open(target_sentence, 'rb')
    train_x = []
    train_y = []
    print "Loading data"
    n = 0
    n_interval = 50000
    liness = source_.readlines(-1)
    linest = target_.readlines(-1)
    print '{} train sentence in total'.format(len(liness))
    bs = 0
    bt = 0
    for ls, lt in zip(liness, linest):
        ss = []
        ls = ls.replace('\r', '')
        ls = ls.replace('\n', '')
        ls = ls + ' </s>'
        words = ls.split(' ')
        for word in words:
            try:
                ss.append(source_vocab[word])
            except:
                ss.append(1)
        st = []
        lt = lt.replace('\r', '')
        lt = lt.replace('\n', '')
        lt = lt + ' </s>'
        words = lt.split(' ')
        for word in words:
            try:
                st.append(target_vocab[word])
            except:
                st.append(1)
        if ls == ' </s>': bs += 1
        if lt == ' </s>': bt += 1
        if ls != ' </s>' and lt != ' </s>':
            train_x.append(ss)
            train_y.append(st)

        n += 1
        if n % n_interval == 0:
            print '        {} solved'.format(n)
    print '{} bad source sentence detected ! {} in total'.format(bs, len(train_x))
    print '{} bad target sentence detected ! {} in total'.format(bt, len(train_x))
    source_sentence_dev = './data/{}-{}-dev.{}.tok'.format(source, target, source)
    target_sentence_dev = './data/{}-{}-dev.{}.tok'.format(source, target, target)
    source_ = open(source_sentence_dev, 'rb')
    target_ = open(target_sentence_dev, 'rb')
    valid_x = []
    valid_y = []
    n = 0
    lines = source_.readlines(-1)
    print '{} sentence of source dev in total'.format(len(lines))
    for line in lines:
        sentence = []
        line = line.replace('\r', '')
        line = line.replace('\n', '')
        line = line + ' </s>'
        words = line.split(' ')
        for word in words:
            try:
                sentence.append(source_vocab[word])
            except:
                sentence.append(1)
        valid_x.append(sentence)
        n += 1
        if n % n_interval == 0:
            print '        {} solved'.format(n)
    n = 0
    lines = target_.readlines(-1)
    print '{} sentence of target dev in total'.format(len(lines))
    for line in lines:
        sentence = []
        line = line.replace('\r', '')
        line = line.replace('\n', '')
        line = line + ' </s>'
        words = line.split(' ')
        for word in words:
            try:
                sentence.append(target_vocab[word])
            except:
                sentence.append(1)
        valid_y.append(sentence)
        n += 1
        if n % n_interval == 0:
            print '        {} solved'.format(n)
    test_x = valid_x
    test_y = valid_y
    return train_x, valid_x, test_x, train_y, valid_y, test_y
