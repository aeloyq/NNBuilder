# -*- coding: utf-8 -*-
"""
Created on  三月 19 2:29 2017

@author: aeloyq
"""
import theano
import theano.tensor as T
import numpy as np
import nnbuilder.layers.lstm, nnbuilder.layers.recurrent

baselayer_lstm = nnbuilder.layers.lstm.get
baselayer_rnn = nnbuilder.layers.recurrent.get
baselayer_gru = nnbuilder.layers.gru.get
from nnbuilder.layers.layers import layer_tools


class get_rnn(baselayer_rnn):
    def __init__(self, in_dim, unit_dim, h_0_init=False, activation=T.tanh, **kwargs):
        baselayer_rnn.__init__(self, in_dim, unit_dim, h_0_init, activation, **kwargs)

    def set_y_mask(self, tvar):
        self.y_mask = tvar

    def get_state_before(self):
        inp = T.sum(self.input * self.x_mask.dimshuffle(0, 1, 'x'), 0)
        self.state_before = T.dot(inp, self.wt) + self.bi

    def get_output(self):
        self.get_state_before()
        self.get_n_samples()
        h_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX),
                      self.n_samples, self.unit_dim)
        if self.h_0_init: h_0 = T.reshape(T.tile(self.h_0, self.n_samples), [self.n_samples, self.unit_dim])
        lin_out, scan_update = theano.scan(self.step, outputs_info=[h_0],
                                           non_sequences=[self.state_before], name=self.name + '_Scan',
                                           n_steps=self.y_mask.shape[0])
        self.output = lin_out
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)

    def step(self, h_, x_):
        h = T.dot(h_, self.u) + x_
        if self.activation is not None:
            h = self.activation(h)
        else:
            h = h
        if self.hidden_unit_dropout:
            if self.ops is not None:
                h = self.ops(h)
        return h

class get_gru_readout(baselayer_gru):
    def __init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim, r_0_init=False,
                 activation=T.tanh,
                 **kwargs):
        baselayer_gru.__init__(self, in_dim, unit_dim, False, activation, **kwargs)
        self.emb_dim = emb_dim
        self.attention_dim = attention_dim
        self.vocab_dim = vocab_dim
        self.r_0_init = r_0_init
        if self.r_0_init:
            self.r_0 = 'r_0'
            self.params.append(self.r_0)
        self.ws = 'ws'
        self.wc = 'wc'
        self.ch='ch'
        self.uo = 'uo'
        self.co = 'co'
        self.bis = 'bis'
        self.bio = 'bio'
        self.s0wt = 's0wt'
        self.s1wt = 's1wt'
        self.s1bi = 's1bi'
        self.param_init_function['ws'] = self.param_init_functions.uniform
        self.param_init_function['wc'] = self.param_init_functions.uniform
        self.param_init_function['ch'] = self.param_init_functions.uniform
        self.param_init_function['uo'] = self.param_init_functions.uniform
        self.param_init_function['co'] = self.param_init_functions.uniform
        self.param_init_function['bis'] = self.param_init_functions.zeros
        self.param_init_function['bio'] = self.param_init_functions.zeros
        self.param_init_function['sowt'] = self.param_init_functions.uniform
        self.param_init_function['s1wt'] = self.param_init_functions.randn
        self.param_init_function['s1bi'] = self.param_init_functions.zeros
        self.params= [self.u,self.bi,self.wc,self.ws,self.bis,self.ug,self.big,self.ch,self.uo,
                       self.co,self.bio,self.s0wt,self.s1wt,self.s1bi]

    def init_layer_params(self):
        baselayer_gru.init_layer_params(self)
        self.ws = theano.shared(value=self.param_init_function['ws'](self.in_dim, self.unit_dim),
                                name='Ws_' + self.name, borrow=True)
        self.wc = theano.shared(value=self.param_init_function['wc'](self.in_dim * 2, self.unit_dim * 2),
                                name='Wc_' + self.name, borrow=True)
        self.ch = theano.shared(value=self.param_init_function['ch'](self.in_dim * 2, self.unit_dim),
                                name='Ch_' + self.name, borrow=True)
        self.uo = theano.shared(value=self.param_init_function['uo'](self.unit_dim, self.unit_dim),
                                name='Uo_' + self.name, borrow=True)
        self.co = theano.shared(value=self.param_init_function['co'](self.in_dim * 2, self.unit_dim),
                                name='Co_' + self.name, borrow=True)
        self.bis = theano.shared(value=self.param_init_function['bis'](self.unit_dim),
                                 name='Bis_' + self.name, borrow=True)
        self.bio = theano.shared(value=self.param_init_function['bio'](self.unit_dim),
                                 name='Bio_' + self.name, borrow=True)
        self.s0wt = theano.shared(value=self.param_init_function['sowt'](self.unit_dim,self.emb_dim),
                                 name='S0wt_' + self.name, borrow=True)
        self.s1wt = theano.shared(value=self.param_init_function['s1wt'](self.emb_dim,self.vocab_dim),
                                 name='S1wt_' + self.name, borrow=True)
        self.s1bi = theano.shared(value=self.param_init_function['s1bi'](self.vocab_dim),
                                 name='S1bi_' + self.name, borrow=True)
        if self.r_0_init:
            r_0_values = np.zeros(self.emb_dim, 'float32')
            self.r_0 = theano.shared(value=r_0_values, name='R_0' + '_' + self.name, borrow=True)
        self.params = [self.u,self.bi,self.wc,self.ws,self.bis,self.ug,self.big,self.ch,self.uo,
                       self.co,self.bio,self.s0wt,self.s1wt,self.s1bi]
        if self.r_0_init: self.params.append(self.r_0)

    def set_y_mask(self, tvar):
        self.y_mask = tvar

    def set_y(self,tvar):
        self.y=tvar


    def get_output(self):
        self.get_n_samples()
        h_0 = T.tanh(T.dot(self.input[0,:,self.in_dim:self.in_dim*2], self.ws) + self.bis)
        ci=T.dot(T.mean(self.input* self.x_mask[:, :, None],0),self.wc)+ self.big
        [ h, y,cost], scan_update = theano.scan(self.step,
                                           sequences=[self.y,self.y_mask],
                                           outputs_info=[ h_0, None,None],
                                           non_sequences=[ci],
                                           name=self.name + '_Scan',
                                           n_steps=self.y_mask.shape[0])
        self.output = y
        self.loss=T.mean(cost)
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)
        self.predict()

    def step(self, y_true,y_m, h_,ci):
        preact = T.dot(h_, self.ug) +ci

        rg = T.nnet.sigmoid(self.slice(preact, 0, self.unit_dim))
        zg = T.nnet.sigmoid(self.slice(preact, 1, self.unit_dim))

        h_c = T.tanh(T.dot(ci, self.ch) + T.dot(rg * h_, self.u) + self.bi)

        h = (1 - zg) * h_ + zg * h_c

        if self.hidden_unit_dropout:
            if self.ops is not None:
                h = self.ops(h)

        t = T.dot(h, self.uo) + T.dot(ci, self.co) + self.bio

        s_0 = T.nnet.softmax(T.dot(t, self.s0wt))
        y = T.nnet.softmax(T.dot(s_0, self.s1wt) + self.s1bi)

        cost=T.nnet.categorical_crossentropy(y,y_true)

        return  h, y,cost

    def predict(self):

        self.pred_Y=T.argmax(self.output,2)

    def cost(self, Y):
        return T.mean(self.loss)

    def error(self,Y):
        return T.mean(T.neq(self.pred_Y,self.y))

class get_gru_readout_feedback(get_gru_readout):
    def __init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim, r_0_init=False,
                 activation=T.tanh,
                 **kwargs):
        get_gru_readout.__init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim, r_0_init,
                 activation, **kwargs)
        self.e = 'e'
        self.vo = 'vo'
        self.param_init_function['e'] = self.param_init_functions.randn
        self.param_init_function['vo'] = self.param_init_functions.zeros
        self.params.extend([self.wt,self.e, self.vo,self.wg])

    def init_layer_params(self):
        get_gru_readout.init_layer_params(self)
        self.wt=theano.shared(value=self.param_init_function['wt'](self.emb_dim, self.unit_dim),
                                name='Wt_' + self.name, borrow=True)
        self.wg = theano.shared(value=self.param_init_function['wg'](self.emb_dim, self.unit_dim * 2),
                                name='Wg_' + self.name, borrow=True)
        self.e = theano.shared(value=self.param_init_function['e'](self.vocab_dim, self.emb_dim),
                               name='E_' + self.name, borrow=True)
        self.vo = theano.shared(value=self.param_init_function['vo'](self.emb_dim, self.unit_dim),
                                name='Vo_' + self.name, borrow=True)
        self.params.extend([self.wt, self.wg,self.e,self.vo])

    def get_output(self):
        self.get_n_samples()
        h_0 = T.tanh(T.dot(self.input[0,:,self.in_dim:self.in_dim*2], self.ws) + self.bis)
        r_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), self.n_samples, self.emb_dim)
        ci = T.dot(T.mean(self.input * self.x_mask[:, :, None], 0), self.wc) + self.big
        if self.r_0_init: r_0 = T.reshape(T.tile(self.r_0, self.n_samples), [self.n_samples, self.emb_dim])
        [r, h, y,cost], scan_update = theano.scan(self.step,
                                           sequences=[self.y,self.y_mask],
                                           outputs_info=[r_0, h_0, None,None],
                                           non_sequences=[ci],
                                           name=self.name + '_Scan',
                                           n_steps=self.y_mask.shape[0])
        self.output = y
        self.loss=cost
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)
        self.predict()
    def step(self, y_true,y_m,r_, h_,ci):

        preact = T.dot(h_, self.ug) + T.dot(r_,self.wg)+ci

        rg = T.nnet.sigmoid(self.slice(preact, 0, self.unit_dim))
        zg = T.nnet.sigmoid(self.slice(preact, 1, self.unit_dim))

        h_c = T.tanh(T.dot(r_, self.wt) +T.dot(ci, self.ch) + T.dot(rg * h_, self.u) + self.bi)

        h = (1 - zg) * h_ + zg * h_c

        if self.hidden_unit_dropout:
            if self.ops is not None:
                h = self.ops(h)

        t = T.dot(h, self.uo)  + T.dot(r_,self.vo)+T.dot(ci, self.co) + self.bio

        s_0=T.nnet.softmax(T.dot(t,self.s0wt))
        y=T.nnet.softmax(T.dot(s_0,self.s1wt)+self.s1bi)

        r = T.dot(y, self.e)

        cost=T.nnet.categorical_crossentropy(y,y_true)
        return r, h, y,cost

class get_gru_maxout_readout_feedback(get_gru_readout_feedback):
    def __init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim, r_0_init=False,
                 activation=T.tanh,
                 **kwargs):
        get_gru_readout_feedback.__init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim, r_0_init,
                 activation, **kwargs)

    def init_layer_params(self):
        get_gru_readout_feedback.init_layer_params(self)
        self.s0wt = theano.shared(value=self.param_init_function['sowt'](self.unit_dim/2,self.emb_dim),
                                 name='S0wt_' + self.name, borrow=True)
        self.params = [self.u,self.bi,self.wc,self.ws,self.bis,self.ug,self.big,self.ch,self.uo,
                       self.co,self.bio,self.s0wt,self.s1wt,self.s1bi,self.wt,self.e, self.vo,self.wg]
        if self.r_0_init: self.params.append(self.r_0)

    def step(self, y_true,y_m,r_, h_,ci):

        preact = T.dot(h_, self.ug) + T.dot(r_,self.wg)+ci

        rg = T.nnet.sigmoid(self.slice(preact, 0, self.unit_dim))
        zg = T.nnet.sigmoid(self.slice(preact, 1, self.unit_dim))

        h_c = T.tanh(T.dot(r_, self.wt) + T.dot(ci, self.ch) + T.dot(rg * h_, self.u) + self.bi)

        h = (1 - zg) * h_ + zg * h_c

        if self.hidden_unit_dropout:
            if self.ops is not None:
                h = self.ops(h)

        t_ = T.dot(h, self.uo)  + T.dot(r_,self.vo)+T.dot(ci, self.co) + self.bio

        output_dim = self.unit_dim // 2
        new_shape = ([t_.shape[i] for i in range(t_.ndim - 1)] + [output_dim, 2])
        t = T.max(t_.reshape(new_shape, ndim=t_.ndim + 1), axis=t_.ndim)

        s_0=T.nnet.softmax(T.dot(t,self.s0wt))
        y=T.nnet.softmax(T.dot(s_0,self.s1wt)+self.s1bi)

        r = T.dot(y, self.e)

        cost=T.nnet.categorical_crossentropy(y,y_true)
        return r, h, y,cost

class get_gru_attention_maxout_readout_feedback(get_gru_maxout_readout_feedback):
    def __init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim, r_0_init=False,
                 activation=T.tanh,
                 **kwargs):
        get_gru_maxout_readout_feedback.__init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim, r_0_init,
                 activation, **kwargs)

        self.wa = 'wa'
        self.ua = 'ua'
        self.wv = 'wv'
        self.bia = 'bia'
        self.param_init_function['wa'] = self.param_init_functions.uniform
        self.param_init_function['ua'] = self.param_init_functions.uniform
        self.param_init_function['wv'] = self.param_init_functions.uniform
        self.param_init_function['bia'] = self.param_init_functions.zeros
        self.params.extend([self.wa, self.ua, self.wv,  self.bia])

    def init_layer_params(self):
        get_gru_maxout_readout_feedback.init_layer_params(self)
        self.wa = theano.shared(value=self.param_init_function['wa'](self.unit_dim, self.attention_dim),
                                name='Wa_' + self.name, borrow=True)
        self.ua = theano.shared(value=self.param_init_function['ua'](self.in_dim * 2, self.attention_dim),
                                name='Ua_' + self.name, borrow=True)
        self.wv = theano.shared(value=self.param_init_function['wv'](self.attention_dim),
                                name='Wv_' + self.name, borrow=True)
        self.bia = theano.shared(value=self.param_init_function['bia'](self.attention_dim),
                                 name='Bia_' + self.name, borrow=True)
        self.params.extend( [self.wa, self.ua, self.wv, self.bia])

    def get_output(self):
        self.get_n_samples()
        h_0 = T.tanh(T.dot(self.input[0,:,self.in_dim:self.in_dim*2], self.ws) + self.bis)
        r_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), self.n_samples, self.emb_dim)
        if self.r_0_init: r_0 = T.reshape(T.tile(self.r_0, self.n_samples), [self.n_samples, self.emb_dim])
        [r, h, y,cost], scan_update = theano.scan(self.step,
                                           sequences=[self.y,self.y_mask],
                                           outputs_info=[r_0, h_0, None,None],
                                           non_sequences=[self.x_mask,self.input],
                                           name=self.name + '_Scan',
                                           n_steps=self.y_mask.shape[0])
        self.output = y
        self.loss=T.mean(cost)
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)
        self.predict()

    def step(self, y_true,y_m, r_, h_, x_m, bs):
        attention = T.exp(T.dot(T.tanh(T.dot(h_, self.wa) +T.dot(bs,self.ua)+self.bia), self.wv)*x_m)
        attention_norm = attention / T.sum(attention, 0)
        ci = T.sum(attention_norm[:,:,None] * bs,0)

        preact = T.dot(h_, self.ug) + T.dot(r_, self.wg) + T.dot(ci,self.wc)+self.big

        rg = T.nnet.sigmoid(self.slice(preact, 0, self.unit_dim))
        zg = T.nnet.sigmoid(self.slice(preact, 1, self.unit_dim))

        h_c = T.tanh(T.dot(r_, self.wt) + T.dot(ci, self.ch) + T.dot(rg * h_, self.u) + self.bi)

        h = (1 - zg) * h_ + zg * h_c

        if self.hidden_unit_dropout:
            if self.ops is not None:
                h = self.ops(h)

        t_ = T.dot(h, self.uo) + T.dot(r_, self.vo) + T.dot(ci, self.co) + self.bio

        output_dim = self.unit_dim // 2
        new_shape = ([t_.shape[i] for i in range(t_.ndim - 1)] +[output_dim, 2])

        t = T.max(t_.reshape(new_shape, ndim=t_.ndim + 1),axis=t_.ndim)

        s_0 = T.nnet.softmax(T.dot(t, self.s0wt))
        y = T.nnet.softmax(T.dot(s_0, self.s1wt) + self.s1bi)

        r = T.dot(y, self.e)

        cost=T.nnet.categorical_crossentropy(y,y_true)*y_m
        return r, h, y,cost

    def sp(self,enc_h, x_mk,dec_s_):
        eij=T.dot(T.tanh(T.dot(dec_s_, self.wa) + T.dot(enc_h, self.ua) + self.bia),self.wv)*x_mk
        return eij


class get_lstm(baselayer_lstm):
    def __init__(self, in_dim, unit_dim, h_0_init=False, c_0_init=False, activation=T.tanh, **kwargs):
        baselayer_lstm.__init__(self, in_dim, unit_dim, h_0_init, c_0_init, activation, **kwargs)
        self.input_way = self.output_ways.final

    def set_y_mask(self, tvar):
        self.y_mask = tvar


    def get_output(self):
        self.get_state_before()
        self.get_n_samples()
        h_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), self.n_samples, self.unit_dim)
        c_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), self.n_samples, self.unit_dim)
        if self.h_0_init: h_0 = T.reshape(T.tile(self.h_0, self.n_samples), [self.n_samples, self.unit_dim])
        if self.c_0_init: c_0 = T.reshape(T.tile(self.c_0, self.n_samples), [self.n_samples, self.unit_dim])
        lin_out, scan_update = theano.scan(self.step, outputs_info=[h_0, c_0],
                                           non_sequences=[self.state_before], name=self.name + '_Scan',
                                           n_steps=self.y_mask.shape[0])
        self.output = lin_out[0]
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)

    def step(self, h_, c_, x_):
        preact = T.dot(h_, self.u) + x_

        i = T.nnet.sigmoid(self.slice(preact, 0, self.unit_dim))
        f = T.nnet.sigmoid(self.slice(preact, 1, self.unit_dim))
        o = T.nnet.sigmoid(self.slice(preact, 2, self.unit_dim))
        c = T.tanh(self.slice(preact, 3, self.unit_dim))

        c = f * c_ + i * c

        if self.cell_unit_dropout:
            if self.ops is not None:
                c = self.ops(c)

        h = o * T.tanh(c)

        if self.hidden_unit_dropout:
            if self.ops is not None:
                h = self.ops(h)

        return [h, c]
class get_lstm_readout(baselayer_lstm):
    def __init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim, c_0_init=False, r_0_init=False,
                 activation=T.tanh,
                 **kwargs):
        baselayer_lstm.__init__(self, in_dim, unit_dim, False, c_0_init, activation, **kwargs)
        self.emb_dim = emb_dim
        self.attention_dim = attention_dim
        self.vocab_dim = vocab_dim
        self.r_0_init = r_0_init
        if self.r_0_init:
            self.r_0 = 'r_0'
            self.params.append(self.r_0)
        self.ws = 'ws'
        self.wv = 'wv'
        self.wc = 'wc'
        self.uo = 'uo'
        self.co = 'co'
        self.bis = 'bis'
        self.bio = 'bio'
        self.s0wt = 's0wt'
        self.s1wt = 's1wt'
        self.s1bi = 's1bi'
        self.param_init_function['ws'] = self.param_init_functions.uniform
        self.param_init_function['wv'] = self.param_init_functions.uniform
        self.param_init_function['wc'] = self.param_init_functions.uniform
        self.param_init_function['uo'] = self.param_init_functions.uniform
        self.param_init_function['co'] = self.param_init_functions.uniform
        self.param_init_function['bis'] = self.param_init_functions.zeros
        self.param_init_function['bio'] = self.param_init_functions.zeros
        self.param_init_function['sowt'] = self.param_init_functions.uniform
        self.param_init_function['s1wt'] = self.param_init_functions.randn
        self.param_init_function['s1bi'] = self.param_init_functions.zeros
        self.params.extend([self.ws,  self.wv, self.wc, self.uo, self.co, self.bis,  self.bio, self.s0wt, self.s1wt, self.s1bi])

    def init_layer_params(self):
        baselayer_lstm.init_layer_params(self)
        self.ws = theano.shared(value=self.param_init_function['ws'](self.in_dim, self.unit_dim),
                                name='Ws_' + self.name, borrow=True)
        self.wv = theano.shared(value=self.param_init_function['wv'](self.attention_dim, 1),
                                name='Wv_' + self.name, borrow=True)
        self.wc = theano.shared(value=self.param_init_function['wc'](self.in_dim * 2, self.unit_dim * 4),
                                name='Wc_' + self.name, borrow=True)
        self.uo = theano.shared(value=self.param_init_function['uo'](self.unit_dim, self.unit_dim),
                                name='Uo_' + self.name, borrow=True)
        self.co = theano.shared(value=self.param_init_function['co'](self.in_dim * 2, self.unit_dim),
                                name='Co_' + self.name, borrow=True)
        self.bis = theano.shared(value=self.param_init_function['bis'](self.unit_dim),
                                 name='Bis_' + self.name, borrow=True)
        self.bio = theano.shared(value=self.param_init_function['bio'](self.unit_dim),
                                 name='Bio_' + self.name, borrow=True)
        self.s0wt = theano.shared(value=self.param_init_function['sowt'](self.unit_dim,self.emb_dim),
                                 name='S0wt_' + self.name, borrow=True)
        self.s1wt = theano.shared(value=self.param_init_function['s1wt'](self.emb_dim,self.vocab_dim),
                                 name='S1wt_' + self.name, borrow=True)
        self.s1bi = theano.shared(value=self.param_init_function['s1bi'](self.vocab_dim),
                                 name='S1bi_' + self.name, borrow=True)
        if self.r_0_init:
            r_0_values = np.zeros(self.emb_dim, 'float32')
            self.r_0 = theano.shared(value=r_0_values, name='R_0' + '_' + self.name, borrow=True)
        self.params = [self.u, self.bi, self.ws, self.wc, self.uo,
                       self.co, self.bis, self.bio, self.s0wt, self.s1wt, self.s1bi]
        if self.r_0_init: self.params.append(self.r_0)

    def set_y_mask(self, tvar):
        self.y_mask = tvar

    def set_y(self,tvar):
        self.y=tvar


    def get_output(self):
        self.get_n_samples()
        h_0 = T.tanh(T.dot(self.input[0,:,self.in_dim:self.in_dim*2], self.ws) + self.bis)
        c_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), self.n_samples, self.unit_dim)
        self.biencoder_hidden = self.input
        if self.c_0_init: c_0 = T.reshape(T.tile(self.c_0, self.n_samples), [self.n_samples, self.unit_dim])
        [ h, c, y,cost], scan_update = theano.scan(self.step,
                                           sequences=[self.y,self.y_mask],
                                           outputs_info=[ h_0, c_0, None,None],
                                           non_sequences=[self.x_mask,self.biencoder_hidden],
                                           name=self.name + '_Scan',
                                           n_steps=self.y_mask.shape[0])
        self.output = y
        self.loss=T.mean(cost)
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)
        self.predict()

    def step(self, y_true,y_m, h_, c_, x_m, bs):
        ci=T.mean(bs* x_m[:, :, None],0)
        preact = T.dot(h_, self.u) + T.dot(ci, self.wc) + self.bi

        i = T.nnet.sigmoid(self.slice(preact, 0, self.unit_dim))
        f = T.nnet.sigmoid(self.slice(preact, 1, self.unit_dim))
        o = T.nnet.sigmoid(self.slice(preact, 2, self.unit_dim))
        c = T.tanh(self.slice(preact, 3, self.unit_dim))

        c = f * c_ + i * c

        if self.cell_unit_dropout:
            if self.ops is not None:
                c = self.ops(c)

        h = o * T.tanh(c)

        if self.hidden_unit_dropout:
            if self.ops is not None:
                h = self.ops(h)

        t = T.dot(h, self.uo) + T.dot(ci, self.co) + self.bio

        y = T.nnet.softmax(T.dot(T.dot(t,self.s0wt),self.s1wt)+self.s1bi)

        cost=T.nnet.categorical_crossentropy(y,y_true)

        return  h, c, y,cost

    def predict(self):

        self.pred_Y=T.argmax(self.output,2)

    def cost(self, Y):
        return T.mean(self.loss)

    def error(self,Y):
        return T.mean(T.neq(self.pred_Y,self.y))
class get_lstm_readout_feedback(get_lstm_readout):
    def __init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim, c_0_init=False, r_0_init=False,
                 activation=T.tanh,
                 **kwargs):
        get_lstm_readout.__init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim, c_0_init, r_0_init,
                 activation, **kwargs)
        self.e = 'e'
        self.vo = 'vo'
        self.param_init_function['e'] = self.param_init_functions.randn
        self.param_init_function['vo'] = self.param_init_functions.zeros
        self.params.extend([self.e, self.vo])

    def init_layer_params(self):
        get_lstm_readout.init_layer_params(self)
        self.wt=theano.shared(value=self.param_init_function['wt'](self.emb_dim, self.unit_dim*4),
                                name='Wt_' + self.name, borrow=True)
        self.e = theano.shared(value=self.param_init_function['e'](self.vocab_dim, self.emb_dim),
                               name='E_' + self.name, borrow=True)
        self.vo = theano.shared(value=self.param_init_function['vo'](self.emb_dim, self.unit_dim),
                                name='Vo_' + self.name, borrow=True)
        self.params.extend([self.wt, self.e,self.vo])

    def get_output(self):
        self.get_n_samples()
        h_0 = T.tanh(T.dot(self.input[0,:,self.in_dim:self.in_dim*2], self.ws) + self.bis)
        c_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), self.n_samples, self.unit_dim)
        r_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), self.n_samples, self.emb_dim)
        ci = T.mean(self.input * self.x_mask[:, :, None], 0)
        if self.c_0_init: c_0 = T.reshape(T.tile(self.c_0, self.n_samples), [self.n_samples, self.unit_dim])
        if self.r_0_init: r_0 = T.reshape(T.tile(self.r_0, self.n_samples), [self.n_samples, self.emb_dim])
        [r, h, c, y,cost], scan_update = theano.scan(self.step,
                                           sequences=[self.y,self.y_mask],
                                           outputs_info=[r_0, h_0, c_0, None,None],
                                           non_sequences=[ci],
                                           name=self.name + '_Scan',
                                           n_steps=self.y_mask.shape[0])
        self.output = y
        self.loss=cost
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)
        self.predict()

    def step(self, y_true,y_m, r_, h_, c_, ci):
        preact = T.dot(r_, self.wt) + T.dot(h_, self.u) + T.dot(ci, self.wc) + self.bi

        i = T.nnet.sigmoid(self.slice(preact, 0, self.unit_dim))
        f = T.nnet.sigmoid(self.slice(preact, 1, self.unit_dim))
        o = T.nnet.sigmoid(self.slice(preact, 2, self.unit_dim))
        c = T.tanh(self.slice(preact, 3, self.unit_dim))

        c = f * c_ + i * c

        if self.cell_unit_dropout:
            if self.ops is not None:
                c = self.ops(c)

        h = o * T.tanh(c)

        if self.hidden_unit_dropout:
            if self.ops is not None:
                h = self.ops(h)

        t = T.dot(h, self.uo)  + T.dot(r_,self.vo)+T.dot(ci, self.co) + self.bio

        s_0=T.nnet.softmax(T.dot(t,self.s0wt))
        y=T.nnet.softmax(T.dot(s_0,self.s1wt)+self.s1bi)

        r = T.dot(y, self.e)

        cost=T.nnet.categorical_crossentropy(y,y_true)
        return r, h, c, y,cost

class get_lstm_maxout_readout_feedback(get_lstm_readout_feedback):
    def __init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim, c_0_init=False, r_0_init=False,
                 activation=T.tanh,
                 **kwargs):
        get_lstm_readout_feedback.__init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim, c_0_init, r_0_init,
                 activation, **kwargs)

    def init_layer_params(self):
        get_lstm_readout_feedback.init_layer_params(self)
        self.s0wt = theano.shared(value=self.param_init_function['sowt'](self.unit_dim/2,self.emb_dim),
                                 name='S0wt_' + self.name, borrow=True)
        self.params = [self.wt, self.u, self.bi, self.ws, self.wc, self.e, self.uo,
                       self.vo,
                       self.co, self.bis, self.bio, self.s0wt, self.s1wt, self.s1bi]
        if self.r_0_init: self.params.append(self.r_0)

    def step(self, y_true,y_m, r_, h_, c_, ci):
        preact = T.dot(r_, self.wt) + T.dot(h_, self.u) + T.dot(ci, self.wc) + self.bi

        i = T.nnet.sigmoid(self.slice(preact, 0, self.unit_dim))
        f = T.nnet.sigmoid(self.slice(preact, 1, self.unit_dim))
        o = T.nnet.sigmoid(self.slice(preact, 2, self.unit_dim))
        c = T.tanh(self.slice(preact, 3, self.unit_dim))

        c = f * c_ + i * c

        if self.cell_unit_dropout:
            if self.ops is not None:
                c = self.ops(c)

        h = o * T.tanh(c)

        if self.hidden_unit_dropout:
            if self.ops is not None:
                h = self.ops(h)

        t_ = T.dot(h, self.uo) + T.dot(r_, self.vo) + T.dot(ci, self.co) + self.bio

        output_dim = self.unit_dim // 2
        new_shape = ([t_.shape[i] for i in range(t_.ndim - 1)] + [output_dim, 2])
        t = T.max(t_.reshape(new_shape, ndim=t_.ndim + 1), axis=t_.ndim)

        s_0=T.nnet.softmax(T.dot(t,self.s0wt))
        y=T.nnet.softmax(T.dot(s_0,self.s1wt)+self.s1bi)

        r = T.dot(y, self.e)

        cost=T.nnet.categorical_crossentropy(y,y_true)
        return r, h, c, y,cost

class get_lstm_attention_maxout_readout_feedback(get_lstm_maxout_readout_feedback):
    def __init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim, c_0_init=False, r_0_init=False,
                 activation=T.tanh,
                 **kwargs):
        get_lstm_maxout_readout_feedback.__init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim, c_0_init, r_0_init,
                 activation, **kwargs)

        self.wa = 'wa'
        self.ua = 'ua'
        self.wv = 'wv'
        self.bia = 'bia'
        self.param_init_function['wa'] = self.param_init_functions.uniform
        self.param_init_function['ua'] = self.param_init_functions.uniform
        self.param_init_function['wv'] = self.param_init_functions.uniform
        self.param_init_function['bia'] = self.param_init_functions.zeros
        self.params.extend([self.wa, self.ua, self.wv,  self.bia])

    def init_layer_params(self):
        get_lstm_maxout_readout_feedback.init_layer_params(self)
        self.wa = theano.shared(value=self.param_init_function['wa'](self.unit_dim, self.attention_dim),
                                name='Wa_' + self.name, borrow=True)
        self.ua = theano.shared(value=self.param_init_function['ua'](self.in_dim * 2, self.attention_dim),
                                name='Ua_' + self.name, borrow=True)
        self.wv = theano.shared(value=self.param_init_function['wv'](self.attention_dim, 1),
                                name='Wv_' + self.name, borrow=True)
        self.bia = theano.shared(value=self.param_init_function['bia'](self.attention_dim),
                                 name='Bia_' + self.name, borrow=True)
        self.params.extend( [self.wa, self.ua, self.wv, self.bia])

    def get_output(self):
        self.get_n_samples()
        h_0 = T.tanh(T.dot(self.input[0,:,self.in_dim:self.in_dim*2], self.ws) + self.bis)
        c_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), self.n_samples, self.unit_dim)
        r_0 = T.alloc(np.asarray(0.).astype(theano.config.floatX), self.n_samples, self.emb_dim)
        state_before = T.dot(self.input, self.ua)+self.bia
        if self.c_0_init: c_0 = T.reshape(T.tile(self.c_0, self.n_samples), [self.n_samples, self.unit_dim])
        if self.r_0_init: r_0 = T.reshape(T.tile(self.r_0, self.n_samples), [self.n_samples, self.emb_dim])
        self.offset=theano.shared(value=np.array(1e-6).astype('float32'),name='Off_set',borrow=True)
        [r, h, c, y,cost], scan_update = theano.scan(self.step,
                                           sequences=[self.y,self.y_mask],
                                           outputs_info=[r_0, h_0, c_0, None,None],
                                           non_sequences=[self.x_mask, state_before,self.input],
                                           name=self.name + '_Scan',
                                           n_steps=self.y_mask.shape[0])
        self.output = y
        self.loss=T.mean(cost)
        if self.output_dropout:
            if self.ops is not None:
                self.output = self.ops(self.output)
        self.predict()

    def step(self, y_true,y_m, r_, h_, c_, x_m, bh,bs):
        attention = T.exp(T.dot((T.dot(h_, self.wa) +bh) * x_m[:, :, None], self.wv))
        attention_norm = (attention / (T.sum(attention, 0)))
        ci = T.sum((attention_norm.reshape([attention_norm.shape[0],attention_norm.shape[1]]))[:, :, None] * bs,0)
        preact = T.dot(r_, self.wt) + T.dot(h_, self.u) + T.dot(ci, self.wc) + self.bi

        i = T.nnet.sigmoid(self.slice(preact, 0, self.unit_dim))
        f = T.nnet.sigmoid(self.slice(preact, 1, self.unit_dim))
        o = T.nnet.sigmoid(self.slice(preact, 2, self.unit_dim))
        c = T.tanh(self.slice(preact, 3, self.unit_dim))

        c = f * c_ + i * c

        if self.cell_unit_dropout:
            if self.ops is not None:
                c = self.ops(c)

        h = o * T.tanh(c)

        if self.hidden_unit_dropout:
            if self.ops is not None:
                h = self.ops(h)

        t_ = T.dot(h, self.uo) + T.dot(r_, self.vo) + T.dot(ci, self.co) + self.bio

        output_dim = self.unit_dim // 2
        new_shape = ([t_.shape[i] for i in range(t_.ndim - 1)] +[output_dim, 2])

        t = T.max(t_.reshape(new_shape, ndim=t_.ndim + 1),axis=t_.ndim)

        s_0 = T.nnet.softmax(T.dot(t, self.s0wt))
        y = T.nnet.softmax(T.dot(s_0, self.s1wt) + self.s1bi)

        r = T.dot(y, self.e)

        cost=T.nnet.categorical_crossentropy(y,y_true)
        #cost=-T.log(y[T.arange(y_true.shape[0]),y_true])*y_m
        return r, h, c, y,cost
