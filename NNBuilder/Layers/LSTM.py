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
    def __init__(self,Rng,N_in,N_hl,Name='undefined',Wt=None,Bi=None,Wt_init='uniform',Bi_init='zeros',Activation=T.tanh):
        Hidden_Layer.__init__(self,Rng,N_in,N_hl,Name,Wt,Bi,Wt_init,Bi_init,Activation)

        class layer(Hidden_Layer):
            def __init__(self, Rng, N_in, N_hl, Name='undefined', Wt=None, Bi=None, U=None, Wt_init='uniform',
                         Bi_init='zeros', U_init='uniform', Activation=T.tanh):
                Hidden_Layer.__init__(self, Rng, N_in, N_hl, Name, Wt, Bi, Wt_init, Bi_init, Activation)
                self.U = U
                self.U_init = U_init

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
                U_values = Layer_Tools.Fully_connected_lstm_U_init(self.Rng, self.N_units, self.U, self.U_init)
                U = theano.shared(value=U_values, name='U' + '_' + self.Name, borrow=True)
                self.U = U
                self.wt_bi_pack()
                self.wt_bi_inited = True

            def wt_bi_pack(self):
                self.params = [self.Wt, self.Bi, self.U]

            def output_func(self):
                def _slice(_x, n, dim):
                    if _x.ndim == 3:
                        return _x[:, :, n * dim:(n + 1) * dim]
                    return _x[:, n * dim:(n + 1) * dim]

                def _step(m_, x_, h_, c_):
                    preact = T.dot(h_, self.U)
                    preact += x_

                    i = T.nnet.sigmoid(_slice(preact, 0, self.N_units))
                    f = T.nnet.sigmoid(_slice(preact, 1, self.N_units))
                    o = T.nnet.sigmoid(_slice(preact, 2, self.N_units))
                    c = T.tanh(_slice(preact, 3, self.N_units))

                    c = f * c_ + i * c
                    c = m_[:, None] * c + (1. - m_)[:, None] * c_

                    h = o * T.tanh(c)
                    h = m_[:, None] * h + (1. - m_)[:, None] * h_

                    return h, c

                if self.Inputs.ndim == 3:
                    n_samples = self.Inputs.shape[1]
                else:
                    n_samples = 1
                lin_out, scan_update = theano.scan(_step, sequences=[self.Inputs,self.mask.T],
                                                   outputs_info=[
                                                       T.alloc(np.asarray(0.).astype(theano.config.floatX),
                                                               n_samples, self.N_units),T.alloc(np.asarray(0.).astype(theano.config.floatX),
                                                               n_samples, self.N_units)], name=self.Name + '_Scan',
                                                   n_steps=self.Inputs.shape[0])
                self.outputs = lin_out[-1]