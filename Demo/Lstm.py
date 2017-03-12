# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""
import sys
sys.path.append('..')
import nnbuilder
import nnbuilder.config as config
from nnbuilder.dataprepares import Load_imdb
from nnbuilder.algrithms import sgd,adadelta
from nnbuilder.extensions import earlystop, monitor,sample,debugmode
from nnbuilder.layers import *
from nnbuilder.model import model
from nnbuilder.mainloop import train
from nnbuilder.visions.Visualization import get_result
from nnbuilder.model import Get_Model_Stream
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
    earlystop.config.valid_freq=370
    sgd.config.learning_rate=0.0001
    data_stream  = Load_imdb(n_words=vocab_size,maxlen=maxlen)
    dim_model=model()
    mask=T.matrix('Mask')
    dim_model.addlayer(embedding.layer(config.rng, N_in=vocab_size, N_dims=word_dim), 'raw', 'Embedding')
    dim_model.addlayer(lstm.layer(config.rng, in_dim=word_dim, N_hl=hidden_dim, out_state='mean_pooling'), 'Embedding', 'LSTM', od_in=mask)
    dim_model.addlayer(softmax.layer(config.rng, in_dim=hidden_dim, N_out=label_dim), 'LSTM', 'Softmax')
    model_stream = Get_Model_Stream(datastream=data_stream, algrithm=adadelta, dim_model=dim_model)  # ,grad_monitor=True)
    sample.config.sample_func = None
    result_stream = train(model_stream=model_stream, datastream=data_stream, extension=[monitor, earlystop, sample])
    vision_return = get_result(result_stream=result_stream, model_stream=model_stream)
