# -*- coding: utf-8 -*-
"""
Created on  三月 19 2:29 2017

@author: aeloyq
"""
import theano
import theano.tensor as T
import numpy as np


from layers import baselayer,layer_tools

class get_lstm_attention(baselayer):
    def __init__(self,in_dim,unit_dim,h_0_init=False,c_0_init=False,r_0_init=False,activation=T.tanh):
        baselayer.__init__(self)
        self.in_dim = in_dim
        self.unit_dim = unit_dim
        self.h_0_init = h_0_init
        self.c_0_init = c_0_init
        self.r_0_init = r_0_init
        self.param_init_function = {'wt': self.param_init_functions.uniform,
                                    'bi': self.param_init_functions.zeros,
                                    'wt_attention':self.param_init_functions.uniform,
                                    'wt_attention_h': self.param_init_functions.uniform,
                                    'bi_attention':self.param_init_functions.zeros,
                                    'wt_readout':self.param_init_functions.uniform,
                                    'bi_readout':self.param_init_functions.zeros,
                                    'u':self.param_init_functions.orthogonal,
                                    'h_0':self.param_init_functions.zeros,
                                    'c_0':self.param_init_functions.zeros,
                                    'r_0': self.param_init_functions.zeros}
        self.wt='wt'
        self.wt_attention='wt_attention'
        self.wt_attention_h = 'wt_attention_h'
        self.u='u'
        self.h_0='h_0'
        self.c_0='c_0'
        self.r_0 = 'c_0'
        self.bi='bi'
        self.bi_attention='bi_attention'
        self.params=[self.wt,self.u,self.wt_attention,self.wt_attention_h]
        if self.h_0_init: self.params.append(self.h_0)
        if self.c_0_init: self.params.append( self.c_0)
        self.params.append(self.bi)
        self.params.append(self.bi_attention)
        self.hidden_unit_dropout = False
        self.cell_unit_dropout=False
        self.output_dropout = False
    def set_input(self,X):
        self.input=X
    def init_layer_params(self):
        wt_values = self.param_init_function['wt'](self.in_dim,self.unit_dim*4)
        bi_values = self.param_init_function['bi'](self.unit_dim*4)
        u_values=self.param_init_function['u'](self.unit_dim,self.unit_dim*4)
        wt_attention_values=self.param_init_function['wt_attention'](self.in_dim,1)
        wt_attention_h_values = self.param_init_function['wt_attention'](self.unit_dim, 1)
        bi_attention_values = self.param_init_function['bi_attention'](1)
        self.wt = theano.shared(value=wt_values, name='Wt'+'_'+self.name, borrow=True)
        self.bi = theano.shared(value=bi_values, name='Bi'+'_'+self.name, borrow=True)
        self.wt_attention = theano.shared(value=wt_attention_values, name='Wt_Attention' + '_' + self.name, borrow=True)
        self.wt_attention_h = theano.shared(value=wt_attention_h_values, name='Wt_Attention_H' + '_' + self.name, borrow=True)
        self.bi_attention = theano.shared(value=bi_attention_values, name='Bi_Attention' + '_' + self.name, borrow=True)
        self.u = theano.shared(value=u_values, name='U' + '_' + self.name, borrow=True)
        if self.h_0_init:
            h_0_values = self.param_init_function['h_0'](self.unit_dim)
            self.h_0 = theano.shared(value=h_0_values, name='H_0' + '_' + self.name, borrow=True)
        if self.c_0_init:
            c_0_values = self.param_init_function['c_0'](self.unit_dim)
            self.c_0 = theano.shared(value=c_0_values, name='C_0' + '_' + self.name, borrow=True)
        self.params = [self.wt, self.u,self.wt_attention,self.wt_attention_h]
        if self.h_0_init: self.params.append( self.h_0)
        if self.c_0_init: self.params.append( self.c_0)
        self.params.append(self.bi)
        self.params.append( self.bi_attention)
    def set_x_mask(self,tvar):
        self.x_mask=tvar
    def set_y_mask(self,tvar):
        self.y_mask=tvar
    def get_output(self):
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step( h_, c_):
            attention_out_h=T.dot(h_,self.wt_attention_h)+ self.bi_attention
            attention_out_h=attention_out_h[:,0]
            attention_out_x = T.dot(self.input, self.wt_attention)
            attention_out_x=attention_out_x[:,:,0]
            attention_out=(attention_out_x+attention_out_h)*self.x_mask
            norm=T.sum(T.transpose(attention_out),1)
            attention_out=attention_out/norm
            preact=T.sum(attention_out.dimshuffle(0,1,'x')*self.input,0)

            preact=T.dot(preact,self.wt)+T.dot(h_,self.u)+self.bi

            i = T.nnet.sigmoid(_slice(preact, 0, self.unit_dim))
            f = T.nnet.sigmoid(_slice(preact, 1, self.unit_dim))
            o = T.nnet.sigmoid(_slice(preact, 2, self.unit_dim))
            c = T.tanh(_slice(preact, 3, self.unit_dim))

            c = f * c_ + i * c

            if self.cell_unit_dropout:
                if self.ops is not None:
                    c = self.ops(c)

            h = o * T.tanh(c)

            if self.hidden_unit_dropout:
                if self.ops is not None:
                    h = self.ops(h)
            return h, c


        if self.input.ndim == 3:
            n_samples = self.input.shape[1]
        else:
            n_samples = 1
        h_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), n_samples, self.unit_dim)
        c_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), n_samples, self.unit_dim)
        if self.h_0_init: h_0 = T.reshape(T.tile(h_0,n_samples),[n_samples,self.unit_dim])
        if self.c_0_init: c_0 = T.reshape(T.tile(c_0,n_samples),[n_samples,self.unit_dim])
        lin_out, scan_update = theano.scan(_step, sequences=[],
                                           outputs_info=[h_0,c_0], name=self.name + '_Scan',
                                           n_steps=self.y_mask.shape[0])
        lin_out = lin_out[0]
        self.output = layer_tools.all(lin_out, self.x_mask)
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)

class get_lstm(baselayer):
    def __init__(self,in_dim,unit_dim,h_0_init=False,c_0_init=False,r_0_init=False,activation=T.tanh):
        baselayer.__init__(self)
        self.in_dim = in_dim
        self.unit_dim = unit_dim
        self.h_0_init = h_0_init
        self.c_0_init = c_0_init
        self.r_0_init = r_0_init
        self.param_init_function = {'wt': self.param_init_functions.uniform,
                                    'bi': self.param_init_functions.zeros,
                                    'u':self.param_init_functions.orthogonal,
                                    'h_0':self.param_init_functions.zeros,
                                    'c_0':self.param_init_functions.zeros}
        self.wt='wt'
        self.u='u'
        self.h_0='h_0'
        self.c_0='c_0'
        self.bi='bi'
        self.params=[self.wt,self.u]
        if self.h_0_init: self.params.append(self.h_0)
        if self.c_0_init: self.params.append( self.c_0)
        self.params.append(self.bi)
        self.hidden_unit_dropout = False
        self.cell_unit_dropout=False
        self.output_dropout = False
    def set_input(self,X):
        self.input=X
    def init_layer_params(self):
        wt_values = self.param_init_function['wt'](self.in_dim,self.unit_dim*4)
        bi_values = self.param_init_function['bi'](self.unit_dim*4)
        u_values=self.param_init_function['u'](self.unit_dim,self.unit_dim*4)
        self.wt = theano.shared(value=wt_values, name='Wt'+'_'+self.name, borrow=True)
        self.bi = theano.shared(value=bi_values, name='Bi'+'_'+self.name, borrow=True)
        self.u = theano.shared(value=u_values, name='U' + '_' + self.name, borrow=True)
        if self.h_0_init:
            h_0_values = self.param_init_function['h_0'](self.unit_dim)
            self.h_0 = theano.shared(value=h_0_values, name='H_0' + '_' + self.name, borrow=True)
        if self.c_0_init:
            c_0_values = self.param_init_function['c_0'](self.unit_dim)
            self.c_0 = theano.shared(value=c_0_values, name='C_0' + '_' + self.name, borrow=True)
        self.params = [self.wt, self.u]
        if self.h_0_init: self.params.append( self.h_0)
        if self.c_0_init: self.params.append( self.c_0)
        self.params.append(self.bi)
    def set_x_mask(self,tvar):
        self.x_mask=tvar
    def set_y_mask(self,tvar):
        self.y_mask=tvar
    def get_output(self):
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        state_before =T.dot(self.input[-1],self.wt)

        def _step( h_, c_):

            preact=state_before+T.dot(h_,self.u)+self.bi

            i = T.nnet.sigmoid(_slice(preact, 0, self.unit_dim))
            f = T.nnet.sigmoid(_slice(preact, 1, self.unit_dim))
            o = T.nnet.sigmoid(_slice(preact, 2, self.unit_dim))
            c = T.tanh(_slice(preact, 3, self.unit_dim))

            c = f * c_ + i * c

            if self.cell_unit_dropout:
                if self.ops is not None:
                    c = self.ops(c)

            h = o * T.tanh(c)

            if self.hidden_unit_dropout:
                if self.ops is not None:
                    h = self.ops(h)
            return h, c


        if self.input.ndim == 3:
            n_samples = self.input.shape[1]
        else:
            n_samples = 1
        h_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), n_samples, self.unit_dim)
        c_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), n_samples, self.unit_dim)
        if self.h_0_init: h_0 = T.reshape(T.tile(h_0,n_samples),[n_samples,self.unit_dim])
        if self.c_0_init: c_0 = T.reshape(T.tile(c_0,n_samples),[n_samples,self.unit_dim])
        lin_out, scan_update = theano.scan(_step, sequences=[],
                                           outputs_info=[h_0,c_0], name=self.name + '_Scan',
                                           n_steps=self.y_mask.shape[0])
        lin_out = lin_out[0]
        self.output = layer_tools.all(lin_out, self.x_mask)
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)



class get_lstm_attention_feedback(baselayer):
    def __init__(self,in_dim,unit_dim,emb_dim,h_0_init=False,c_0_init=False,r_0_init=False,activation=T.tanh):
        baselayer.__init__(self)
        self.in_dim = in_dim
        self.unit_dim = unit_dim
        self.emb_dim=emb_dim
        self.h_0_init = h_0_init
        self.c_0_init = c_0_init
        self.r_0_init = r_0_init
        self.param_init_function = {'wt': self.param_init_functions.uniform,
                                    'bi': self.param_init_functions.zeros,
                                    'wt_attention':self.param_init_functions.uniform,
                                    'bi_attention':self.param_init_functions.zeros,
                                    'wt_readout':self.param_init_functions.uniform,
                                    'bi_readout':self.param_init_functions.zeros,
                                    'u':self.param_init_functions.orthogonal,
                                    'h_0':self.param_init_functions.zeros,
                                    'c_0':self.param_init_functions.zeros,
                                    'r_0': self.param_init_functions.zeros}
        self.wt='wt'
        self.wt_attention='wt_attention'
        self.wt_readout='wt_readout'
        self.u='u'
        self.h_0='h_0'
        self.c_0='c_0'
        self.r_0 = 'c_0'
        self.bi='bi'
        self.bi_attention='bi_attention'
        self.bi_readout='bi_readout'
        self.params=[self.wt,self.u,self.wt_attention,self.wt_readout]
        if self.h_0_init: self.params.append(self.h_0)
        if self.c_0_init: self.params.append( self.c_0)
        if self.r_0_init: self.params.append( self.r_0)
        self.params.append(self.bi)
        self.params.append(self.bi_attention)
        self.params.append(self.bi_readout)
        self.hidden_unit_dropout = True
        self.cell_unit_dropout=True
        self.output_dropout = False
    def set_input(self,X):
        self.input=X
    def init_layer_params(self):
        wt_values = self.param_init_function['wt'](self.in_dim+self.emb_dim,self.unit_dim*4)
        bi_values = self.param_init_function['bi'](self.unit_dim*4)
        u_values=self.param_init_function['u'](self.unit_dim,self.unit_dim*4)
        wt_attention_values=self.param_init_function['wt_attention'](self.in_dim+self.unit_dim,1)
        wt_readout_values = self.param_init_function['wt_readout'](self.unit_dim,self.emb_dim)
        bi_attention_values = self.param_init_function['bi_attention'](1)
        bi_readout_values=self.param_init_function['bi_attention'](self.emb_dim)
        self.wt = theano.shared(value=wt_values, name='Wt'+'_'+self.name, borrow=True)
        self.bi = theano.shared(value=bi_values, name='Bi'+'_'+self.name, borrow=True)
        self.wt_attention = theano.shared(value=wt_attention_values, name='Wt_Attention' + '_' + self.name, borrow=True)
        self.wt_readout = theano.shared(value=wt_readout_values, name='Wt_Readout' + '_' + self.name, borrow=True)
        self.bi_attention = theano.shared(value=bi_attention_values, name='Bi_Attention' + '_' + self.name, borrow=True)
        self.bi_readout = theano.shared(value=bi_readout_values, name='Bi_Readout' + '_' + self.name, borrow=True)
        self.u = theano.shared(value=u_values, name='U' + '_' + self.name, borrow=True)
        if self.h_0_init:
            h_0_values = self.param_init_function['h_0'](self.unit_dim)
            self.h_0 = theano.shared(value=h_0_values, name='H_0' + '_' + self.name, borrow=True)
        if self.c_0_init:
            c_0_values = self.param_init_function['c_0'](self.unit_dim)
            self.c_0 = theano.shared(value=c_0_values, name='C_0' + '_' + self.name, borrow=True)
        if self.r_0_init:
            r_0_values = self.param_init_function['r_0'](self.unit_dim)
            self.r_0 = theano.shared(value=r_0_values, name='R_0' + '_' + self.name, borrow=True)
        self.params = [self.wt, self.u,self.wt_attention,self.wt_readout]
        if self.h_0_init: self.params.append( self.h_0)
        if self.c_0_init: self.params.append( self.c_0)
        if self.r_0_init: self.params.append( self.r_0)
        self.params.append(self.bi)
        self.params.append( self.bi_attention)
        self.params.append(self.bi_readout)
    def set_x_mask(self,tvar):
        self.x_mask=tvar
    def set_y_mask(self,tvar):
        self.y_mask=tvar
    def get_output(self):
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n * dim:(n + 1) * dim]
            return _x[:, n * dim:(n + 1) * dim]

        def _step( h_, c_,r_):
            def _step_attention(x__):
                out_ = T.exp(T.dot(T.concatenate([x__,h_],1), self.wt_attention) + self.bi_attention)
                return out_
            attention_out,attention_update=theano.scan(_step_attention,sequences=[self.input],n_steps=self.input.shape[0])
            attention_out=T.reshape(attention_out,[attention_out.shape[0],attention_out.shape[1]])*self.x_mask
            norm=T.sum(T.transpose(attention_out),1)
            attention_out=attention_out/norm
            preact=T.sum(attention_out.dimshuffle(0,1,'x')*self.input,0)

            preact=T.dot(T.concatenate([preact,r_],1),self.wt)+T.dot(h_,self.u)+self.bi

            i = T.nnet.sigmoid(_slice(preact, 0, self.unit_dim))
            f = T.nnet.sigmoid(_slice(preact, 1, self.unit_dim))
            o = T.nnet.sigmoid(_slice(preact, 2, self.unit_dim))
            c = T.tanh(_slice(preact, 3, self.unit_dim))

            c = f * c_ + i * c

            if self.cell_unit_dropout:
                if self.ops is not None:
                    c = self.ops(c)

            h = o * T.tanh(c)

            if self.hidden_unit_dropout:
                if self.ops is not None:
                    h = self.ops(h)
            r=T.dot(h,self.wt_readout)+self.bi_readout
            return h, c, r


        if self.input.ndim == 3:
            n_samples = self.input.shape[1]
        else:
            n_samples = 1
        h_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), n_samples, self.unit_dim)
        c_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), n_samples, self.unit_dim)
        r_0=T.alloc(np.asarray(0.).astype(theano.config.floatX), n_samples, self.emb_dim)
        if self.h_0_init: h_0 = T.reshape(T.tile(h_0,n_samples),[n_samples,self.unit_dim])
        if self.c_0_init: c_0 = T.reshape(T.tile(c_0,n_samples),[n_samples,self.unit_dim])
        if self.r_0_init: r_0 =T.reshape(T.tile(r_0,n_samples),[n_samples,self.unit_dim])
        lin_out, scan_update = theano.scan(_step, sequences=[],
                                           outputs_info=[h_0,c_0,r_0], name=self.name + '_Scan',
                                           n_steps=self.y_mask.shape[0])
        lin_out=lin_out[2]
        self.output=layer_tools.all(lin_out,self.x_mask)
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)


