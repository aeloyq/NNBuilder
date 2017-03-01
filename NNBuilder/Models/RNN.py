# -*- coding: utf-8 -*-
"""
Created on  Feb 13 11:16 PM 2017

@author: aeloyq
"""
import theano.tensor as T
def get_model(n_in_word_dim,hl_rnn,n_out_word_dim):
    n_hl_rnn_units=hl_rnn[0]
    n_rnn_hl=hl_rnn[1]
    struct =[]
    struct.append( [T.tensor3('X3'),'setx'])
    struct.append([T.matrix('Y2'), 'sety'])
    last_hl='raw'
    last_in=n_in_word_dim
    mask=T.matrix('Mask')
    for n in range(n_rnn_hl):
        struct.append([(['Recurrent',(last_in,n_hl_rnn_units)],last_hl,'Recurrent_HiddenLayer_%d'%(n+1),1,mask),'layer'])
        last_hl='Recurrent_HiddenLayer_%d'%(n+1)
        last_in=n_hl_rnn_units
    struct.append([(['Logistic', (n_hl_rnn_units, n_out_word_dim)], last_hl, 'ReadoutLayer'), 'layer'])
    return struct