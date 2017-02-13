# -*- coding: utf-8 -*-
"""
Created on  Feb 13 11:16 PM 2017

@author: aeloyq
"""
def get_model(n_in,n_out,n_hl,n_hl_units):
    struct = []
    last_hl='raw'
    last_in=n_in
    for n in range(n_hl):
        struct.append([('HiddenLayerFF',(last_in,n_hl_units),last_hl,'HiddenLayer_%d'%(n+1)),'layer'])
        last_hl='HiddenLayer_%d'%(n+1)
        last_in=n_hl_units
    struct.append([('Logistic',(n_hl_units,n_out),last_hl,'OutputLayer'),'layer'])
    return struct