# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import sys
sys.path.append('..')
import NNBuilder
import NNBuilder.config as config
from NNBuilder.DataPrepares import Load_imdb
from NNBuilder.Algrithms import SGD,Adadelta
from NNBuilder.Extensions import Earlystop, Monitor,Sample,DebugMode
from NNBuilder.Layers import *
from NNBuilder.Model import model
from NNBuilder.MainLoop import Train
from NNBuilder.Visions.Visualization import get_result
from NNBuilder.Model import Get_Model_Stream
import theano.tensor as T


if __name__ == '__main__':
    global data_stream, model_stream, result_stream, vision_return
    vocab_size=100000
    label_dim=2
    word_dim=hidden_dim=128
    maxlen=100
    config.batch_size=16
    config.max_epoches=5000
    config.data_path='./Datasets/imdb.pkl'
    config.trans_input=True
    Earlystop.config.valid_freq=370
    SGD.config.learning_rate=0.0001
    data_stream  = Load_imdb(n_words=vocab_size,maxlen=maxlen)
    dim_model=model()
    mask=T.matrix('Mask')
    dim_model.addlayer(Embedding.layer(config.rng,N_in=vocab_size,N_dims=word_dim),'raw','Embedding')
    dim_model.addlayer(LSTM.layer(config.rng,N_in=word_dim,N_hl=hidden_dim,out_state='mean_pooling'),'Embedding','LSTM',od_in=mask)
    dim_model.addlayer(Softmax.layer(config.rng,N_in=hidden_dim,N_out=label_dim),'LSTM','Softmax')
    model_stream = Get_Model_Stream(datastream=data_stream, algrithm=Adadelta, dim_model=dim_model)  # ,grad_monitor=True)
    Sample.config.sample_func = None
    result_stream = Train(model_stream=model_stream, datastream=data_stream, extension=[Monitor, Earlystop, Sample])
    vision_return = get_result(result_stream=result_stream, model_stream=model_stream)
