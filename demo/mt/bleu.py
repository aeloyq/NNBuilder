# -*- coding: utf-8 -*-
"""
Created on  三月 18 19:58 2017

@author: aeloyq
"""

import nnbuilder
from nnbuilder.data import *
from nnbuilder.layers.simple import *
from nnbuilder.layers.sequential import *
from nnbuilder.algrithms import *
from nnbuilder.extensions import *
from nnbuilder.model import *
from nnbuilder.main import *
import dictionary


saveload.config.save_freq=500

source_vocab_size=30000
target_vocab_size=30000

source_emb_dim=620
target_emb_dim=620

enc_dim=1000
dec_dim=1000

config.vocab_source='./data/vocab.en-fr.en.pkl'
config.vocab_target='./data/vocab.en-fr.fr.pkl'

config.name='mt_demo'
config.data_path='./data/devsets.npz'
config.batch_size=80
config.valid_batch_size=64
config.max_epoches=1000
config.savelog=True
config.transpose_x=True
config.transpose_y=True
config.mask_x=True
config.mask_y=True
config.int_x=True
config.int_y=True



model=model(source_vocab_size,Int2dX,Int2dY)
model.sequential(Float2dMask,Float2dMask)
model.add(embedding(source_emb_dim))
model.add(encoder(enc_dim))
#model.add(dropout(0.8))
model.add(decoder(dec_dim,target_emb_dim,target_vocab_size))
#model.add(dropout(0.8))

datas=Load_mt(sort_by_len=False)
model.build()
saveload.config.load_npz(model,'dev')


n=1
model.output.gen_sample(n)
print 'compiling'
fs=theano.function(model.inputs,model.output.sample, on_unused_input='ignore',
                                updates=model.raw_updates)
print 'compile ok'

s_list=[]
s_text=''

mbs=get_minibatches_idx(datas)
mbs=mbs[0]

print 'bleuing'
import progressbar
bar = progressbar.ProgressBar()
for idx,index in bar(mbs):
    data = prepare_data(datas[0], datas[3], index)
    ss = fs(*data)
    for s in ss:
        st=dictionary.mt_bleu(s)
        s_list.append(st)
        s_text+=st+'\r\n'

print 'bleu ok'

print 'dumping'
f=open('./{}.txt'.format(str(n)),'wb')
f.write(s_text+"\r\n")
f.close()
print 'dump ok'