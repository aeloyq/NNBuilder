# -*- coding: utf-8 -*-
"""
Created on  Feb 16 1:28 AM 2017

@author: aeloyq
"""


import numpy as np
import theano
import theano.tensor as T
from Layers import Hidden_Layer,Layer_Tools

''' setup hidden layer of feedforward network inherited from Hidden_Layer '''

class layer(Hidden_Layer):
    def __init__(self,Rng,N_in,N_hl,Name='undefined',Wt=None,Bi=None,U=None,Wt_init='uniform',Bi_init='zeros',U_init='uniform',Activation=T.tanh):
        Hidden_Layer.__init__(self,Rng,N_in,N_hl,Name,Wt,Bi,Wt_init,Bi_init,Activation)
        self.U=U
        self.U_init=U_init
    def init_wt_bi(self):
        if not self.wt_bi_inited:
            Wt_values, Bi_values = Layer_Tools.Fully_connected_weights_init(self.Rng, self.N_in, self.N_units,
                                                                            self.Wt, self.Bi, self.Wt_init,
                                                                            self.Bi_init)
            Wt = theano.shared(value=Wt_values, name='Wt' + '_' + self.Name, borrow=True)
            Bi = theano.shared(value=Bi_values, name='Bi' + '_' + self.Name, borrow=True)
            self.init_u()
            self.Wt, self.Bi = Wt, Bi
            self.wt_bi_pack()
            self.wt_bi_inited = True
    def init_u(self):
        U_values=Layer_Tools.Fully_connected_U_init(self.Rng,self.N_units,self.U,self.U_init)
        U = theano.shared(value=U_values, name='U' + '_' + self.Name, borrow=True)
        self.U = U
        self.wt_bi_pack()
        self.wt_bi_inited = True
    def wt_bi_pack(self):
        self.params=[self.Wt,self.Bi,self.U]
    def add_inputs(self,tvar):
        self.mask=tvar
    def output_func(self):
        def _step(x_,m_,h_):
            out_=T.dot(h_,self.U)+T.dot(x_,self.Wt)+self.Bi
            if self.Activation is not None:
                out_ = self.Activation(out_)
            else:
                out_ = out_
            out_=m_[:, None] * out_+ (1. - m_)[:, None] * out_
            return out_
        if self.Inputs.ndim == 3:
            n_samples = self.Inputs.shape[1]
        else:
            n_samples = 1
        lin_out,scan_update=theano.scan(_step,sequences=[self.Inputs,self.mask],
                                        outputs_info=[
                            T.alloc(np.asarray(0.).astype(theano.config.floatX),
                                    n_samples,self.N_units)],name=self.Name+'_Scan',n_steps=self.Inputs.shape[0])
        self.outputs=lin_out[-1]