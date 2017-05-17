# -*- coding: utf-8 -*-
"""
Created on  三月 16 18:59 2017

@author: aeloyq
"""
import tarfile
import os
import subprocess
import nnbuilder.config as config
import numpy as np
import cPickle as cp

datasets= './datasets/'
scripts_path='./scripts/'
data_path= './data/'
toked_file='tok/'
vocab_file='voc/'
train_path='train/'
valid_path='valid/'
test_path='test/'
source_language='en'
target_language='fr'
trainning_data_tar_file='training-parallel-nc-v10.tgz'
valid_data_tar_file='dev-v2.tgz'
nonbreaking_prefix='nonbreaking_prefix.'
preprocess_file = 'preprocess.py'
tokenizer_file = './scripts/tokenizer.perl'

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

def extract_tar_file_to(file_to_extract, extract_into, source,target):
    extracted_filenames = []
    try:
        print 'extracting file...'
        tar = tarfile.open(file_to_extract, 'r')
        src_trg_files = [ff for ff in tar.getnames() if ff.find(target+'-'+source)!=-1]
        if not len(src_trg_files):
            print("[{}] pair does not exist in the archive!".format(src_trg_files))
        for item in tar:
            # extract only source-target pair
            if item.name in src_trg_files:
                file_path = os.path.join(extract_into, item)
                if not os.path.exists(file_path):
                    print("...extracting [{}] into [{}]"
                                .format(item.name, file_path))
                    tar.extract(item, extract_into)
                else:
                    print("...file exists [{}]".format(file_path))
                extracted_filenames.append(
                    os.path.join(extract_into, item.path))
    except Exception as e:
        print str(e)
    return extracted_filenames

def tokenize_text_files(files_to_tokenize, tokenizer,OUTPUT_DIR):
    name=files_to_tokenize
    print ("Tokenizing file [{}]".format(name))
    out_file = os.path.join(
        OUTPUT_DIR, os.path.basename(name) + '.tok')
    print("...writing tokenized file [{}]".format(out_file))
    var = ["perl", tokenizer,  "-l", name.split('.')[-1]]

    if not os.path.exists(out_file):
        os.remove(out_file)
    else:
        with open(name, 'r') as inp:
            with open(out_file, 'w', 0) as out:
                subprocess.check_call(
                    var, stdin=inp, stdout=out, shell=False)
                print("wrote tokenized file [{}]".format(out_file))


def create_vocabularies( preprocess_file,OUTPUT_DIR,source_vocab,target_vocab):
    src_vocab_name = os.path.join(
        OUTPUT_DIR, 'vocab.{}-{}.{}.pkl'.format(
            source_language, target_language, source_language))
    trg_vocab_name = os.path.join(
        OUTPUT_DIR, 'vocab.{}-{}.{}.pkl'.format(
            source_language, target_language, target_language))
    src_filename = source_language+'-'+target_language+'.'+source_language+'.tok'
    trg_filename = source_language+'-'+target_language+'.'+target_language+'.tok'
    print("Creating source vocabulary [{}]".format(src_vocab_name))
    if not os.path.exists(src_vocab_name):
        subprocess.check_call(" python {} -d {} -v {} {}".format(
            preprocess_file, src_vocab_name, source_vocab,
            os.path.join(OUTPUT_DIR, src_filename)),
            shell=True)
    else:
        print("...file exists [{}]".format(src_vocab_name))

    print("Creating target vocabulary [{}]".format(trg_vocab_name))
    if not os.path.exists(trg_vocab_name):
        subprocess.check_call(" python {} -d {} -v {} {}".format(
            preprocess_file, trg_vocab_name, target_vocab,
            os.path.join(OUTPUT_DIR, trg_filename)),
            shell=True)
    else:
        print("...file exists [{}]".format(trg_vocab_name))
    return src_filename, trg_filename

file_name=source_language+'-'+target_language

#extract_tar_file_to(data_path+trainning_data_tar_file,datasets_path,source_language,target_language)

#tokenize_text_files(data_path +'tmp/'+ file_name+"."+source_language, tokenizer_file, data_path)
#tokenize_text_files(data_path +'tmp/'+ file_name+"."+target_language, tokenizer_file, data_path)
#tokenize_text_files(data_path +'tmp/dev/'+ file_name+'-dev'+"."+source_language, tokenizer_file, data_path)
#tokenize_text_files(data_path +'tmp/dev/'+ file_name+'-dev'+"."+target_language, tokenizer_file, data_path)

#create_vocabularies(preprocess_file,data_path,30000,30000)
config.source='en'
config.target='fr'
datastream=get_data_stream()
#np.savez('./data/datasets.npz',datastream)
train_x, valid_x, test_x, train_y, valid_y, test_y=datastream
np.savez('./data/devsets.npz',[valid_x, valid_x, test_x, valid_y, valid_y, test_y])

