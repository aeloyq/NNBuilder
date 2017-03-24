# -*- coding: utf-8 -*-
"""
Created on  三月 18 19:58 2017

@author: aeloyq
"""


from nnbuilder import config
import numpy as np
import theano
import theano.tensor as T
from nnbuilder.layers import embedding,lstm,decoder,softmax
from nnbuilder.algrithms import sgd,adadelta,rmsprop
from nnbuilder.dataprepares import Load_mt
from nnbuilder.model import model
from nnbuilder.extensions import monitor ,debugmode,saveload,earlystop
from nnbuilder.mainloop import train

source_vocab_size=80000
target_vocab_size=80000

source_emb_dim=512
target_emb_dim=512

enc_dim=1024
dec_dim=1024

vocab_source='./data/vocab.en-fr.en.pkl'
vocab_target='./data/vocab.en-fr.fr.pkl'

config.name='mt_demo'
config.data_path='./data/datasets.npz'
config.batch_size=128
config.valid_batch_size=256
config.max_epoches=1000
config.savelog=True
config.transpose_x=True
config.transpose_y=True
config.mask_x=True
config.mask_y=True
config.int_x=True
config.int_y=True

monitor.config.report_iter_frequence=10
monitor.config.report_iter=True



data=Load_mt()


X=T.imatrix('X')
Y=T.imatrix('Y')
X_mask=T.matrix('X_Mask')
Y_mask=T.matrix('Y_Mask')

emb=embedding.get(in_dim=source_vocab_size,emb_dim=source_emb_dim)
enc=lstm.get_bi(in_dim=source_emb_dim,unit_dim=enc_dim)
enc.set_mask(X_mask)
dec=decoder.get_lstm_attention(in_dim=enc_dim,unit_dim=dec_dim,
                               emb_dim=target_emb_dim)
dec.set_x_mask(X_mask)
dec.set_y_mask(Y_mask)
out=softmax.get_sequence(in_dim=target_emb_dim,unit_dim=target_vocab_size)
out.set_mask(Y_mask)


mt_model=model()
mt_model.set_inputs([X,Y,X_mask,Y_mask])
mt_model.addlayer(emb,X,'emb')
mt_model.addlayer(enc,emb,'enc')
mt_model.addlayer(dec,enc,'dec')
mt_model.addlayer(out,dec,'out')

result=train(datastream=data,model=mt_model,algrithm=sgd,extension=[saveload,monitor,earlystop])


