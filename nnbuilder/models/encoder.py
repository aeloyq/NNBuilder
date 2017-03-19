# -*- coding: utf-8 -*-
"""
Created on  三月 18 22:10 2017

@author: aeloyq
"""
import theano
import theano.tensor as T

from nnbuilder.layers import lstm,gru,recurrent,embedding
from model import base
class model(base):
    def __init__(self,vocab_size,wordvector_dim,recurent_layer,recurent_unit_dim):
        base.__init__()
        self.vocab_size=vocab_size
        self.wordvector_dim=wordvector_dim
        self.recurent_layer=recurent_layer
        self.recurent_unit_dim=recurent_unit_dim
    def add_graph(self,model):
        emb=embedding.get_new(in_dim=self.vocab_size,emb_dim=self.wordvector_dim)
        recurrent_forward=self.recurent_layer.get_new(in_dim=self.wordvector_dim,unit_dim=self.recurent_unit_dim)
        recurrent_backward=self.recurent_layer.get_new(in_dim=self.wordvector_dim,unit_dim=self.recurent_unit_dim)

        model.addlayer(emb,model.X,'embedding_layer')
        model.addlayer(recurrent_forward,emb,'recurrent_forward',model.X_mask)
        model.addlayer(recurrent_backward,T. model.X[:,::], 'recurrent_backward', model.X_mask)
