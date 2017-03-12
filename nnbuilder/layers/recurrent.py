# -*- coding: utf-8 -*-
"""
Created on  Feb 16 1:28 AM 2017

@author: aeloyq
"""


import numpy as np
import theano
import theano.tensor as T
from layers import hidden_layer,layer_tools

''' setup hidden layer of feedforward network inherited from Hidden_Layer '''

class get_new(hidden_layer):
    class output_ways:
        final=layer_tools.final
        all=layer_tools.all
        mean_pooling=layer_tools.mean_pooling

    def __init__(self, in_dim, unit_dim, activation=T.tanh,**kwargs):
        hidden_layer.__init__(self, in_dim, unit_dim,activation,**kwargs)
        self.param_init_function = {'wt': self.param_init_functions.uniform, 'bi': self.param_init_functions.zeros,
                                    'u': self.param_init_functions.uniform}
        self.u = 'u'
        self.params = [self.wt, self.bi, self.u]
        self.output_way = self.output_ways.final
        self.hidden_unit_dropout=True
        self.output_dropout = False

    def init_layer_params(self):
        wt_values = self.param_init_function['wt'](self.in_dim,self.unit_dim)
        bi_values = self.param_init_function['bi'](self.unit_dim)
        u_values=self.param_init_function['u'](self.unit_dim,self.unit_dim)
        self.wt = theano.shared(value=wt_values, name='Wt'+'_'+self.name, borrow=True)
        self.bi = theano.shared(value=bi_values, name='Bi'+'_'+self.name, borrow=True)
        self.u = theano.shared(value=u_values, name='U' + '_' + self.name, borrow=True)
        self.params = [self.wt, self.bi, self.u]

    def set_mask(self,tvar):
        self.mask=tvar

    def get_output(self):
        def _step(x_,m_,h_):
            out_=T.dot(h_,self.u)+T.dot(x_, self.wt)+ self.bi
            if self.activation is not None:
                out_ = self.activation(out_)
            else:
                out_ = out_
            out_=m_[:, None] * out_+ (1. - m_)[:, None] * out_
            if self.hidden_unit_dropout:
                if self.ops is not None:
                    out_=self.ops(out_)
            return out_
        if self.input.ndim == 3:
            n_samples = self.input.shape[1]
        else:
            n_samples = 1
        lin_out,scan_update=theano.scan(_step, sequences=[self.input,self.mask],
                                        outputs_info=[
                            T.alloc(np.asarray(0.).astype(theano.config.floatX),
                                    n_samples, self.unit_dim)], name=self.name + '_Scan', n_steps=self.input.shape[0])
        self.output = self.output_way(lin_out,self.mask)
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)