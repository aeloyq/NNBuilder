# -*- coding: utf-8 -*-
"""
Created on  三月 17 20:20 2017

@author: aeloyq
"""
import nnbuilder.config as config
import  nnbuilder.logger as logger
import cPickle as cp



def get_data_stream():
    source = config.source
    target = config.target
    source_sentence='./data/{}-{}.{}.tok'.format(source,target,source)
    target_sentence = './data/{}-{}.{}.tok'.format(source, target, target)
    source_vocab='./data/vocab.{}-{}.{}.pkl'.format(source,target,source)
    target_vocab = './data/vocab.{}-{}.{}.pkl'.format(source, target, target)
    source_vocab=cp.load(open(source_vocab,'rb'))
    target_vocab = cp.load(open(target_vocab, 'rb'))
    source_=open(source_sentence,'rb')
    target_=open(target_sentence,'rb')
    train_x=[]
    train_y=[]
    print "Loading data"
    n=0
    n_interval=50000
    lines=source_.readlines(-1)
    print '{} sentence of source in total'.format(len(lines))
    for line in lines:
        sentence=[]
        line=line.replace('\r','')
        line=line.replace('\n', '')
        line=line+' </s>'
        words=line.split(' ')
        for word in words:
            try:
                sentence.append(source_vocab[word])
            except:
                sentence.append(1)
        train_x.append(sentence)
        n+=1
        if n%n_interval==0:
            print '        {} solved'.format(n)
    n=0
    lines = target_.readlines(-1)
    print '{} sentence of target in total'.format(len(lines))
    for line in lines:
        sentence = []
        line =line.replace('\r', '')
        line =line.replace('\n', '')
        line = line+' </s>'
        words = line.split(' ')
        for word in words:
            try:
                sentence.append(target_vocab[word])
            except:
                sentence.append(1)
        train_y.append(sentence)
        n += 1
        if n % n_interval == 0:
            print '        {} solved'.format(n)
    source_sentence_dev = './data/{}-{}-dev.{}.tok'.format(source, target, source)
    target_sentence_dev = './data/{}-{}-dev.{}.tok'.format(source, target, target)
    source_ = open(source_sentence_dev, 'rb')
    target_ = open(target_sentence_dev, 'rb')
    valid_x=[]
    valid_y=[]
    n = 0
    lines = source_.readlines(-1)
    print '{} sentence of source dev in total'.format(len(lines))
    for line in lines:
        sentence=[]
        line =line.replace('\r','')
        line =line.replace('\n', '')
        line=line+' </s>'
        words=line.split(' ')
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
        line =line.replace('\r', '')
        line =line.replace('\n', '')
        line = line+' </s>'
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
    test_x=valid_x
    test_y=valid_y
    return train_x,valid_x,test_x,train_y,valid_y,test_y