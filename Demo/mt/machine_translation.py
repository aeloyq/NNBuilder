# -*- coding: utf-8 -*-
"""
Created on  三月 18 19:58 2017

@author: aeloyq
"""


from nnbuilder import config
import numpy as np
import theano
import theano.tensor as T
from nnbuilder.layers import embedding
from nnbuilder.models import encoder,decoder
from nnbuilder.algrithms import sgd,adadelta,rmsprop
from nnbuilder.dataprepares import Load_mt
from nnbuilder.model import model
from nnbuilder.extensions import monitor ,debugmode,saveload,earlystop,sample
from nnbuilder.mainloop import train
import dictionary

#theano.config.profile=True
#theano.config.profile_memory=True
#theano.config.optimizer='fast_compile'
debugmode.config.debug_time=3
debugmode.config.debug_batch=5

sample.config.sample_freq=20
sample.config.sample_times=2
sample.config.sample_func=dictionary.mt_sample

saveload.config.save_freq=500

source_vocab_size=40000
target_vocab_size=40000

source_emb_dim=512
target_emb_dim=512

enc_dim=1024
dec_dim=1024

config.vocab_source='./data/vocab.en-fr.en.pkl'
config.vocab_target='./data/vocab.en-fr.fr.pkl'

sgd.config.learning_rate=0.7
sgd.config.grad_clip_norm=1.
sgd.config.if_clip=True

config.name='mt_demo'
config.data_path='./data/datasets.npz'
config.batch_size=40
config.valid_batch_size=64
config.max_epoches=1000
config.savelog=True
config.transpose_x=True
config.transpose_y=True
config.mask_x=True
config.mask_y=True
config.int_x=True
config.int_y=True

monitor.config.report_iter_frequence=1
monitor.config.report_iter=True


X=T.imatrix('X')
Y=T.imatrix('Y')
X_mask=T.matrix('X_Mask')
Y_mask=T.matrix('Y_Mask')

emb=embedding.get(in_dim=source_vocab_size,emb_dim=source_emb_dim)
enc=encoder.get_bi_gru_(in_dim=source_emb_dim,unit_dim=enc_dim)
enc.set_x_mask(X_mask)
dec=decoder.get_gru_attention_maxout_readout_feedback(in_dim=enc_dim,unit_dim=dec_dim,
                                                       attention_dim=1024,
                                                       emb_dim=target_emb_dim,
                                                       vocab_dim=target_vocab_size)
dec.set_x_mask(X_mask)
dec.set_y_mask(Y_mask)
dec.set_y(Y)


mt_model=model()
mt_model.set_inputs([X,Y,X_mask,Y_mask])
mt_model.addlayer(emb,X,'emb')
mt_model.addlayer(enc,emb,'enc')
mt_model.addlayer(dec,enc,'dec')

data=Load_mt(maxlen=50,sort_by_len=False)

result=train(datastream=data,model=mt_model,algrithm=sgd,extension=[monitor,sample])


