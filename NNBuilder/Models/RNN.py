# -*- coding: utf-8 -*-
"""
Created on  Feb 13 11:16 PM 2017

@author: aeloyq
"""
def get_model(n_in_vcab_length,n_in_word_dim,hl_rnn,n_out_word_dim,n_out_vcab_length):
    n_hl_rnn_units=hl_rnn[0]
    n_rnn_hl=hl_rnn[1]
    struct = []
    last_hl='raw'
    struct.append([(['Embedding',(n_in_vcab_length,n_in_word_dim)],last_hl,'word2vec'),'layer'])
    last_hl = 'word2vec'
    last_in=n_in_word_dim
    for n in range(n_rnn_hl):
        struct.append([(['Recurrent',(last_in,n_hl_rnn_units)],last_hl,'Recurrent_HiddenLayer_%d'%(n+1)),'layer'])
        last_hl='Recurrent_HiddenLayer_%d'%(n+1)
        last_in=n_hl_rnn_units
    struct.append([(['Readout', (n_hl_rnn_units, n_out_word_dim)], last_hl, 'ReadoutLayer'), 'layer'])
    struct.append([(['Softmax',(n_out_word_dim,n_out_vcab_length)],'ReadoutLayer','OutputLayer'),'layer'])
    return struct