# -*- coding: utf-8 -*-
"""
Created on  四月 26 21:08 2017

@author: aeloyq
"""
import numpy as np
import theano
import theano.tensor as T
from basic import *
from simple import *
from utils import *
from roles import *
from ops import *


class sequential(baselayer):
    def __init__(self, **kwargs):
        baselayer.__init__(self, **kwargs)
        self.mask = True
        self.go_backwards = False
        self.x_mask = None
        self.y_mask = None
        self.out = 'all'
        self.setattr('out')
        self.setattr('mask')
        self.setattr('y_mask')
        self.setattr('x_mask')
        self.setattr('go_backwards')

    def get_n_samples(self):
        self.n_samples = self.input.shape[-2]

    def prepare(self, X, P):
        self.state_before = [X]
        if self.mask: self.state_before = [X, self.x_mask]
        self.initiate_state = None
        self.context = None
        self.n_steps = self.state_before[0].shape[0]

    def get_context(self):
        if self.out == 'final':
            self.output = self.output[-1]
        elif self.out == 'all':
            self.output = self.output
        elif self.out == 'mean':
            if not self.mask:
                self.output = self.output.mean(self.output.ndim - 1)
            else:
                self.output = (self.output * self.x_mask[:, :, None]).sum(0) / self.x_mask.sum(0)[:, None]

    def get_output(self, X, P):
        self.get_n_samples()
        self.prepare(X, P)
        step = self.apply(X, P)
        self.output, updates = self.scan(step)
        self.get_context()
        self.updates.update(updates)

    def add_mask(self, tvar, mask):
        if tvar.ndim == 2:
            return mask[:, None] * tvar + (1. - mask)[:, None] * tvar
        elif tvar.ndim == 3:
            return mask[None, :, None] * tvar + (1. - mask)[None, :, None] * tvar
        elif tvar.ndim == 4:
            return mask[None, None, :, None] * tvar + (1. - mask)[None, None, :, None] * tvar

    def apply(self, X, P):
        def step(x):
            h = x
            h = self.addops('hidden_unit', h, dropout)
            return h

        def step_mask(x, m):
            h = x
            h = self.add_mask(h, m)
            h = self.addops('hidden_unit', h, dropout)
            return h

        if not self.mask:
            return step
        else:
            return step_mask

    def scan(self, step):
        return theano.scan(step, sequences=self.state_before, outputs_info=self.initiate_state,
                           non_sequences=self.context, n_steps=self.n_steps, go_backwards=self.go_backwards,
                           strict=True)

    def feedstep(self):
        return self.apply(None, None)

    def feedscan(self, X=None, P=None):
        if X is None:
            X = self.input
        if P == None or P == {}:
            P = self.P
        else:
            if self.children_name in P:
                dict = self.P
                P = dict.update(P[self.children_name])
        self.set_input(X)
        self.get_output(X, P)
        self.merge(1)
        return self.output

    def slice(self, _x, n, dim):
        if _x.ndim == 2:
            return _x[:, n * dim:(n + 1) * dim]
        elif _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        elif _x.ndim == 4:
            return _x[:, :, :, n * dim:(n + 1) * dim]


class rnn(sequential):
    def __init__(self, unit, mask=True, activation=T.tanh, **kwargs):
        sequential.__init__(self, mask=mask, **kwargs)
        self.mask = mask
        self.unit_dim = unit
        self.activation = activation

    def set_children(self):
        self.children['lb'] = linear_bias(self.unit_dim)

    def init_params(self):
        self.u = self.allocate(orthogonal, 'U', weight, self.unit_dim, self.unit_dim)

    def prepare(self, X, P):
        input = self.children['lb'].feedforward(X)
        self.state_before = [input]
        if self.mask:
            self.state_before.append(self.x_mask)
        self.initiate_state = [T.zeros([self.n_samples, self.unit_dim], theano.config.floatX)]
        self.context = [P['U']]
        self.n_steps = input.shape[0]

    def apply(self, X, P):
        def step(x, h_, u):
            h = T.dot(h_, u) + x
            if self.activation is not None:
                h = self.activation(h)
            h = self.addops('hidden_unit', h, dropout)
            return h

        def step_mask(x, m, h_, u):
            h = T.dot(h_, u) + x
            if self.activation is not None:
                h = self.activation(h)
            h = self.add_mask(h, m)
            h = self.addops('hidden_unit', h, dropout)
            return h

        if not self.mask:
            return step
        else:
            return step_mask


class gru(sequential):
    def __init__(self, unit, mask=True, **kwargs):
        sequential.__init__(self, mask=mask, **kwargs)
        self.mask = mask
        self.out = 'all'
        self.unit_dim = unit
        self.condition = None
        self.setattr('out')
        self.setattr('condition')

    def set_children(self):
        self.children['input'] = linear_bias(self.unit_dim)
        self.children['gate'] = linear_bias(self.unit_dim * 2)

    def init_params(self):
        self.u = self.allocate(orthogonal, 'U', weight, self.unit_dim, self.unit_dim)
        self.ug = self.allocate(orthogonal, 'Ug', weight, self.unit_dim, self.unit_dim * 2)

    def prepare(self, X, P):
        self.state_before = [self.children['input'].feedforward(X),
                             self.children['gate'].feedforward(X)]
        if self.mask:
            self.state_before.append(self.x_mask)
        self.initiate_state = [T.zeros([self.n_samples, self.unit_dim], theano.config.floatX)]
        self.context = [P['U'], P['Ug']]
        self.n_steps = self.input.shape[0]

    def apply(self, X, P):
        def step(x, xg, h_, u, ug):
            gate = xg + T.dot(h_, ug)
            gate = self.conditional(gate)
            gate = T.nnet.sigmoid(gate)
            z_gate = self.slice(gate, 0, self.unit_dim)
            z_gate = self.addops('z_gate', z_gate, dropout, False)
            r_gate = self.slice(gate, 1, self.unit_dim)
            r_gate = self.addops('r_gate', r_gate, dropout, False)
            h_c = T.tanh(x + T.dot(r_gate * h_, u))
            h = (1 - z_gate) * h_ + z_gate * h_c
            h = self.addops('hidden_unit', h, dropout)
            return h

        def step_mask(x, xg, m, h_, u, ug):
            gate = xg + T.dot(h_, ug)
            gate = self.conditional(gate)
            gate = T.nnet.sigmoid(gate)
            z_gate = self.slice(gate, 0, self.unit_dim)
            z_gate = self.addops('z_gate', z_gate, dropout, False)
            r_gate = self.slice(gate, 1, self.unit_dim)
            r_gate = self.addops('r_gate', r_gate, dropout, False)
            h_c = T.tanh(x + T.dot(r_gate * h_, u))
            h = (1 - z_gate) * h_ + z_gate * h_c
            h = self.add_mask(h, m)
            h = self.addops('hidden_unit', h, dropout)
            return h

        if not self.mask:
            return step
        else:
            return step_mask

    def conditional(self, gate):
        if self.condition is None:
            return gate
        else:
            return self.condition()


class lstm(sequential):
    def __init__(self, unit, mask=True, **kwargs):
        sequential.__init__(self, mask=mask, **kwargs)
        self.mask = mask
        self.out = 'all'
        self.unit_dim = unit
        self.condition = None
        self.setattr('out')
        self.setattr('condition')

    def set_children(self):
        self.children['input'] = linear_bias(self.unit_dim * 4)

    def init_params(self):
        self.u = self.allocate(orthogonal, 'U', weight, self.unit_dim, self.unit_dim * 4)

    def prepare(self, X, P):
        self.state_before = [self.children['input'].feedforward(X)]
        if self.mask:
            self.state_before.append(self.x_mask)
        self.initiate_state = [T.zeros([self.n_samples, self.unit_dim], theano.config.floatX),
                               T.zeros([self.n_samples, self.unit_dim], theano.config.floatX)]
        self.context = [P['U']]
        self.n_steps = self.input.shape[0]

    def get_output(self, X, P):
        self.get_n_samples()
        self.prepare(X, P)
        step = self.apply(X, P)
        self.output, updates = self.scan(step)
        self.output = self.output[0]
        self.get_context()
        self.updates.update(updates)

    def apply(self, X, P):
        def step(x, h_, c_, u):
            gate = x + T.dot(h_, u)
            gate = self.conditional(gate, h_)
            gate = T.nnet.sigmoid(gate)
            f_gate = self.slice(gate, 0, self.unit_dim)
            f_gate = self.addops('f_gate', f_gate, dropout, False)
            i_gate = self.slice(gate, 1, self.unit_dim)
            i_gate = self.addops('i_gate', i_gate, dropout, False)
            o_gate = self.slice(gate, 2, self.unit_dim)
            o_gate = self.addops('o_gate', o_gate, dropout, False)
            cell = self.slice(gate, 3, self.unit_dim)
            c = f_gate * c_ + i_gate * cell
            c = self.addops('cell_unit', c, dropout)
            h = o_gate * T.nnet.sigmoid(c)
            h = self.addops('hidden_unit', h, dropout)
            return h, c

        def step_mask(x, m, h_, c_, u):
            gate = x + T.dot(h_, u)
            gate = self.conditional(gate, h_)
            gate = T.nnet.sigmoid(gate)
            f_gate = self.slice(gate, 0, self.unit_dim)
            f_gate = self.addops('f_gate', f_gate, dropout, False)
            i_gate = self.slice(gate, 1, self.unit_dim)
            i_gate = self.addops('i_gate', i_gate, dropout, False)
            o_gate = self.slice(gate, 2, self.unit_dim)
            o_gate = self.addops('o_gate', o_gate, dropout, False)
            cell = self.slice(gate, 3, self.unit_dim)
            c = f_gate * c_ + i_gate * cell
            c = self.add_mask(c, m)
            c = self.addops('cell_unit', c, dropout)
            h = o_gate * T.nnet.sigmoid(c)
            h = self.add_mask(h, m)
            h = self.addops('hidden_unit', h, dropout)
            return h, c

        if not self.mask:
            return step
        else:
            return step_mask

    def conditional(self, gate, state_before):
        if self.condition is None:
            return gate
        else:
            return gate + self.condition(state_before)


class encoder(sequential):
    def __init__(self, unit, core=gru, structure='bi', **kwargs):
        sequential.__init__(self, **kwargs)
        self.unit_dim = unit
        self.core = core
        self.structure = structure

    def set_children(self):
        if self.structure == 'single':
            self.children['forward'] = self.core(self.unit_dim, self.mask)
        elif self.structure == 'bi':
            self.children['forward'] = self.core(self.unit_dim, self.mask)
            self.children['backward'] = self.core(self.unit_dim, self.mask, go_backwards=True)

    def get_output(self, X, P):
        self.output = self.apply(X, P)

    def apply(self, X, P):
        if self.structure == 'single':
            return self.children['forward'].feedscan(X, P)
        elif self.structure == 'bi':
            fwd = self.children['forward'].feedscan(X, P)
            bwd = self.children['backward'].feedscan(X, P)
            return concatenate([fwd, bwd[::-1]], fwd.ndim - 1)


class attention(layer):
    def __init__(self, unit, s_dim, o_dim=1, **kwargs):
        layer.__init__(self, unit, **kwargs)
        self.s_dim = s_dim
        self.o_dim = o_dim

    def set_children(self):
        if self.o_dim != 1:
            self.children['dec_s'] = linear(self.unit_dim * 2, in_dim=self.s_dim)
        else:
            self.children['dec_s'] = linear(self.unit_dim, in_dim=self.s_dim)
        self.children['combine'] = linear_bias(self.o_dim, in_dim=self.unit_dim)

    def apply(self, X, P):
        s_ = X[0]
        pctx = X[1]
        mask = X[2]
        if self.o_dim == 1:
            att_layer_1 = T.tanh(pctx + self.children['dec_s'].feedforward(s_, P))
            att_layer_1 = self.addops('att_in', att_layer_1, dropout)
            att_layer_2 = self.children['combine'].feedforward(att_layer_1, P)
            shp = []
            for i in range(att_layer_2.ndim - 1):
                shp.append(att_layer_2.shape[i])
            att = att_layer_2.reshape(shp)

            eij = T.exp(att)
            if eij.ndim == 3:
                eij = eij * mask[:, :,None]
            else:
                eij = eij * mask
            aij = eij / eij.sum(0, keepdims=True)
            return aij
        else:
            pctx1 = self.slice(pctx, 0, self.unit_dim)
            pctx2 = self.slice(pctx, 1, self.unit_dim)
            patt = self.children['dec_s'].feedforward(s_, P)
            att1 = self.slice(patt, 0, self.unit_dim)
            att2 = self.slice(patt, 1, self.unit_dim)
            att_layer_1=T.tanh(pctx1+att1)*T.nnet.sigmoid(pctx2+att2)
            att_layer_2 = self.children['combine'].feedforward(att_layer_1, P)
            att = att_layer_2
            if att.ndim == 4:
                eij = T.exp(att) * mask[:, :,None, None]
            else:
                eij = T.exp(att) * mask[:, :, None]
            aij = eij / eij.sum(0, keepdims=True)
            return aij

    def slice(self, _x, n, dim):
        if _x.ndim == 2:
            return _x[:, n * dim:(n + 1) * dim]
        elif _x.ndim == 3:
            return _x[:, :, n * dim:(n + 1) * dim]
        elif _x.ndim == 4:
            return _x[:, :, :, n * dim:(n + 1) * dim]


class emitter(layer):
    def __init__(self, unit, h_dim, s_dim, emb_dim, is_maxout=True, **kwargs):
        layer.__init__(self, unit, **kwargs)
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.emb_dim = emb_dim
        self.maxout = is_maxout

    def set_children(self):
        self.children['recurent'] = linear_bias(self.in_dim, in_dim=self.s_dim)
        self.children['peek'] = linear(self.in_dim, in_dim=self.emb_dim)
        self.children['glimpse'] = linear(self.in_dim, in_dim=self.h_dim)
        in_dim = self.in_dim
        if self.maxout:
            self.children['maxout'] = maxout(2)
            in_dim = in_dim / 2
        self.children['predict'] = std(self.unit_dim, in_dim=in_dim)

    def apply(self, X, P):
        s = X[0]
        y_ = X[1]
        ctx = X[2]
        prob = self.children['recurent'].feedforward(s, P) + self.children['peek'].feedforward(y_, P) + self.children[
            'glimpse'].feedforward(ctx, P)
        if self.maxout:
            prob = self.children['maxout'].feedforward(prob, P)
        else:
            prob = T.tanh(prob)
        prob = self.addops('emit_gate', prob, dropout)
        prob = self.children['predict'].feedforward(prob, P)
        emit_word = 0
        shape = prob.shape
        if prob.ndim == 2:
            emit_word = T.nnet.softmax(prob)
        elif prob.ndim == 3:
            emit_word = T.nnet.softmax(prob.reshape([shape[0] * shape[1], shape[2]]))
        elif prob.ndim == 4:
            emit_word = T.nnet.softmax(prob.reshape([shape[0] * shape[1] * shape[2], shape[3]]))
        return emit_word


class decoder(sequential):
    def __init__(self, unit, emb_dim, vocab_size, core=gru,
                 state_initiate='mean', be=2, is_maxout=True, is_ma=False, **kwargs):
        sequential.__init__(self, **kwargs)
        self.unit_dim = unit
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.core = core
        self.be = be
        self.state_initiate = state_initiate
        self.attention_unit = unit
        self.maxout = is_maxout
        self.ma = is_ma
        self.y = None
        self.setattr('beam_size')
        self.setattr('attention_unit')

    def set_children(self):
        self.children['state_dec'] = self.core(self.unit_dim, mask=False, in_dim=self.emb_dim)
        self.children['glimpse_dec'] = self.core(self.unit_dim, mask=False, in_dim=self.in_dim * self.be)
        self.children['emitter'] = emitter(self.vocab_size, self.in_dim * self.be, self.unit_dim, self.emb_dim,
                                           in_dim=self.unit_dim, is_maxout=self.maxout)
        if not self.ma:
            self.children['context'] = linear_bias(self.attention_unit, in_dim=self.in_dim * self.be)
            self.children['attention'] = attention(self.attention_unit, self.unit_dim)
        else:
            self.children['context'] = linear_bias(self.attention_unit * 2, in_dim=self.in_dim * self.be)
            self.children['attention'] = attention(self.attention_unit, self.unit_dim,self.in_dim * self.be)
        self.children['peek'] = lookuptable(self.emb_dim, in_dim=self.vocab_size)

    def init_params(self):
        self.wt_iniate_s = self.allocate(uniform, 'Wt_iniate_s', weight, self.in_dim, self.unit_dim)

    def prepare(self, X, P):
        self.state_before = self.children['peek'].feedforward(self.y)
        emb_shifted = T.zeros_like(self.state_before)
        emb_shifted = T.set_subtensor(emb_shifted[1:], self.state_before[:-1])
        self.y_ = emb_shifted
        y_ = self.children['state_dec'].children['input'].feedforward(emb_shifted)
        self.state_before = [y_]
        if self.core == gru:
            yg_ = self.children['state_dec'].children['gate'].feedforward(emb_shifted)
            self.state_before = [y_, yg_]

        mean_ctx = (self.input[:, :, -self.in_dim:] * self.x_mask[:, :, None]).sum(0) / self.x_mask.sum(0)[:, None]
        s_0 = mean_ctx
        if self.state_initiate == 'final': s_0 = self.input[0, :, -self.in_dim:]
        s_0 = T.tanh(T.dot(s_0, self.wt_iniate_s))

        self.initiate_state = [s_0]
        self.s_0=s_0
        if isinstance(self.core, lstm):
            self.initiate_state.append(T.zeros([self.n_samples, self.unit_dim], theano.config.floatX))
        self.initiate_state.append(None)

        self.context = [self.children['context'].feedforward(self.input), self.input]
        self.context.append(self.x_mask)
        plist = []
        if self.core == gru:
            plist = ['U_state_dec', 'Ug_state_dec', 'Wt_attention_dec_s', 'Wt_attention_combine',
                     'Bi_attention_combine', 'Wt_glimpse_dec_input', 'Bi_glimpse_dec_input',
                     'Wt_glimpse_dec_gate', 'Bi_glimpse_dec_gate', 'U_glimpse_dec', 'Ug_glimpse_dec']
        elif self.core == lstm:
            plist = ['U_state_dec', 'Wt_attention_dec_s', 'Wt_attention_combine', 'Bi_attention_combine',
                     'Wt_glimpse_dec_input', 'Bi_glimpse_dec_input', 'U_glimpse_dec']
        for i in plist:
            ii = i.split('_')
            iii = ii[0] + '_' + self.name
            for j in range(1, len(ii)):
                iii = iii + '_' + ii[j]
            self.context.append(self.params[iii])

        self.n_steps = self.y.shape[0]

    def get_output(self, X, P):
        self.get_n_samples()
        self.prepare(X, P)
        step = self.apply(X, P)
        [self.s, self.c], updates = self.scan(step)
        o = self.children['emitter'].feedforward([self.s, self.y_, self.c])
        self.output_cost = o
        self.output = o.reshape([self.y.shape[0], self.n_samples, self.vocab_size])
        self.updates.update(updates)

    def apply(self, X, P):
        def step_lstm(y_, s_, c_, pctx, ctx, x_m, su, adwt, acwt, acbi, gwt, gbi, gu):
            s1, c1 = self.children['state_dec'].feedstep()(y_, s_, c_, su)
            aij = self.children['attention'].feedforward([s1, pctx, x_m],
                                                         {'dec_s': {'Wt': adwt}, 'combine': {'Wt': acwt, 'Bi': acbi}})
            if not self.ma:
                if aij.ndim==2:
                    ci = (ctx * aij[:, :, None]).sum(0)
                elif aij.ndim==3:
                    ci = (ctx[:,:,None,:] * aij[:,:, :, None]).sum(0)
            else:
                ci = (ctx * aij).sum(0)

            condition = T.dot(gwt, ci) + gbi
            s2, c2 = self.children['glimpse_dec'].feedstep()(condition, s1, c1, gu)
            s = s2
            return s, ci

        def step_gru(y_, yg_, s_, pctx, ctx, x_m, su, sug, adwt, acwt, acbi, gwt, gbi, gwtg, gbig, gu, gug):
            s1 = self.children['state_dec'].feedstep()(y_, yg_, s_, su, sug)
            aij = self.children['attention'].feedforward([s1, pctx, x_m], {
                'attention': {'dec_s': {'Wt': adwt}, 'combine': {'Wt': acwt, 'Bi': acbi}}})


            if not self.ma:
                if aij.ndim==2:
                    ci = (ctx * aij[:, :, None]).sum(0)
                elif aij.ndim==3:

                    ci = (ctx[:,:,None,:] * aij[:,:, :, None]).sum(0)
            else:
                ci = (ctx * aij).sum(0)

            condition = T.dot(ci, gwt) + gbi
            conditiong = T.dot(ci, gwtg) + gbig
            s2 = self.children['glimpse_dec'].feedstep()(condition, conditiong, s1, gu, gug)
            s = s2
            return s, ci

        if self.core == lstm:
            return step_lstm
        elif self.core == gru:
            return step_gru

    def scan(self, step):
        return theano.scan(step, sequences=self.state_before, outputs_info=self.initiate_state,
                           non_sequences=self.context, n_steps=self.n_steps, strict=True)

    def get_predict(self):
        y_0 = T.zeros([self.n_samples, self.emb_dim])
        s_0 = self.s_0
        initiate_state = [y_0, s_0]
        if isinstance(self.core, lstm):
            initiate_state.append(T.zeros([self.n_samples, self.unit_dim], theano.config.floatX))
        initiate_state.append(None)

        context = [self.children['context'].feedforward(self.input), self.input]
        context.append(self.x_mask)
        plist = []
        if self.core == gru:
            plist = ['Wt_state_dec_input', 'Bi_state_dec_input', 'Wt_state_dec_gate', 'Bi_state_dec_gate',
                     'U_state_dec', 'Ug_state_dec', 'Wt_attention_dec_s', 'Wt_attention_combine',
                     'Bi_attention_combine', 'Wt_glimpse_dec_input', 'Bi_glimpse_dec_input',
                     'Wt_glimpse_dec_gate', 'Bi_glimpse_dec_gate', 'U_glimpse_dec', 'Ug_glimpse_dec',
                     'Wt_emitter_recurent', 'Bi_emitter_recurent', 'Wt_emitter_peek', 'Wt_emitter_glimpse',
                     'Wt_emitter_predict', 'Wemb_peek']
        elif self.core == lstm:
            plist = ['Wt_state_dec_input', 'Bi_state_dec_input', 'U_state_dec', 'Wt_attention_dec_s',
                     'Wt_attention_combine', 'Bi_attention_combine',
                     'Wt_glimpse_dec_input', 'Bi_glimpse_dec_input', 'U_glimpse_dec',
                     'Wt_emitter_recurent', 'Bi_emitter_recurent', 'Wt_emitter_peek', 'Wt_emitter_glimpse',
                     'Wt_emitter_predict', 'Wemb_peek']
        for i in plist:
            ii = i.split('_')
            iii = ii[0] + '_' + self.name
            for j in range(1, len(ii)):
                iii = iii + '_' + ii[j]
            context.append(self.params[iii])

        def step_lstm(y_emb_, s_,c_, pctx, ctx, x_m, swt, sbi, su, adwt, acwt, acbi, gwt, gbi, gu, erwt, erbi,
                      epwt, egwt, edwt, wemb):
            yi = T.dot(y_emb_, swt) + sbi
            s,c, ci = self.feedstep()(yi, s_,c_, pctx, ctx, x_m, su, adwt, acwt, acbi, gwt, gbi, gu)

            prob = self.children['emitter'].feedforward([s, y_emb_, ci], {
                'emitter': {'recurent': {'Wt': erwt, 'Bi': erbi}, 'peek': {'Wt': epwt}, 'glimpse': {'Wt': egwt},
                            'predict': {'Wt': edwt}}})
            prob = self.trng.multinomial(pvals=prob)
            pred = prob.argmax(-1)
            y_emb = T.reshape(wemb[pred], [self.n_samples, self.emb_dim])
            return [y_emb, s,c, pred]

        def step_gru(y_emb_, s_, pctx, ctx, x_m, swt, sbi, swtg, sbig, su, sug, adwt, acwt, acbi, gwt,
                     gbi, gwtg,
                     gbig, gu, gug, erwt, erbi, epwt, egwt, edwt, wemb):
            yi = T.dot(y_emb_, swt) + sbi
            ygi = T.dot(y_emb_, swtg) + sbig
            s, ci = self.feedstep()(yi, ygi, s_, pctx, ctx, x_m, su, sug, adwt, acwt, acbi, gwt, gbi, gwtg, gbig, gu,
                                    gug)

            prob = self.children['emitter'].feedforward([s, y_emb_, ci], {
                'emitter': {'recurent': {'Wt': erwt, 'Bi': erbi}, 'peek': {'Wt': epwt}, 'glimpse': {'Wt': egwt},
                            'predict': {'Wt': edwt}}})
            prob = self.trng.multinomial(pvals=prob)
            pred = prob.argmax(-1)
            y_emb = T.reshape(wemb[pred], [self.n_samples, self.emb_dim])
            return [y_emb, s, pred]

        step = None
        if self.core == lstm:
            step = step_lstm
        elif self.core == gru:
            step = step_gru

        result, updates = theano.scan(step, sequences=[], outputs_info=initiate_state,
                                      non_sequences=context, strict=True, n_steps=self.y.shape[0])
        updates.update(self.updates)

        self.raw_updates = updates

        pred = 0

        if self.core == lstm:
            y_emb, s, c, pred = result
        elif self.core == gru:
            y_emb, s, pred = result

        self.predict = pred

    def gen_sample(self,beam_size=12):
        self.beam_size=beam_size
        y_0 = T.zeros([self.n_samples,self.beam_size,  self.emb_dim])
        y_mm_0 = T.ones([self.n_samples, self.beam_size])
        if self.beam_size==1:
            y_mm_0 = T.unbroadcast(y_mm_0,1)
        s_0 = self.s_0
        s_0 = T.tile(s_0, self.beam_size).reshape([self.beam_size, self.n_samples, self.unit_dim])
        s_0 =s_0.dimshuffle(1,0,2)
        initiate_state = [y_0, s_0]
        if isinstance(self.core, lstm):
            initiate_state.append(T.zeros([self.n_samples,self.beam_size,  self.unit_dim], theano.config.floatX))
        initiate_state.extend([y_mm_0,T.constant(1),None,None,None,None])

        pctx=self.children['context'].feedforward(self.input)[:,:,None,:]
        ctx=self.input
        context = [pctx, ctx,self.x_mask]
        plist = []
        if self.core == gru:
            plist = ['Wt_state_dec_input', 'Bi_state_dec_input', 'Wt_state_dec_gate', 'Bi_state_dec_gate',
                     'U_state_dec', 'Ug_state_dec', 'Wt_attention_dec_s', 'Wt_attention_combine',
                     'Bi_attention_combine', 'Wt_glimpse_dec_input', 'Bi_glimpse_dec_input',
                     'Wt_glimpse_dec_gate', 'Bi_glimpse_dec_gate', 'U_glimpse_dec', 'Ug_glimpse_dec',
                     'Wt_emitter_recurent', 'Bi_emitter_recurent', 'Wt_emitter_peek', 'Wt_emitter_glimpse',
                     'Wt_emitter_predict', 'Wemb_peek']
        elif self.core == lstm:
            plist = ['Wt_state_dec_input', 'Bi_state_dec_input', 'U_state_dec', 'Wt_attention_dec_s',
                     'Wt_attention_combine', 'Bi_attention_combine',
                     'Wt_glimpse_dec_input', 'Bi_glimpse_dec_input', 'U_glimpse_dec',
                     'Wt_emitter_recurent', 'Bi_emitter_recurent', 'Wt_emitter_peek', 'Wt_emitter_glimpse',
                     'Wt_emitter_predict', 'Wemb_peek']
        for i in plist:
            ii = i.split('_')
            iii = ii[0] + '_' + self.name
            for j in range(1, len(ii)):
                iii = iii + '_' + ii[j]
            context.append(self.params[iii])

        from theano.ifelse import ifelse

        def slice(_x, n, dim):
            return _x[:, :, n * dim:(n + 1) * dim]

        def step_lstm(y_emb_, s_, c_, y_mm, pctx, ctx, x_m, swt, sbi, su, adwt, acwt, acbi, gwt, gbi, gu, erwt, erbi,
                      epwt, egwt, edwt, wemb):
            yi = T.dot(y_emb_, swt) + sbi
            s1, c1 = self.children['state_dec'].feedstep()[0](yi, s_, c_, su)
            aij = self.children['attention'].feedforward([s1, pctx, x_m], {
                'attention': {'dec_s': {'Wt': adwt}, 'combine': {'Wt': acwt, 'Bi': acbi}}})
            ci = (ctx * aij[:, :, None]).sum(0)
            condition = T.dot(ci, gwt) + gbi
            s2, c2 = self.children['glimpse_dec'].feedstep()[0](condition, s1, c1, gu)
            s = s2
            c = c2
            prob = T.dot(s, erwt) + T.dot(y_emb_, epwt) + T.dot(ci, egwt) + erbi
            if self.maxout:
                prob = self.children['emitter'].children['maxout'].feedforward(prob)
            else:
                prob = T.tanh(prob)
            prob = T.dot(prob, edwt)
            prob = prob.dimshuffle(1, 0, 2)
            prob = prob * y_mm[:, :, None]
            prob_flat = prob.reshape([prob.shape[0], prob.shape[1] * prob.shape[2]])
            y = T.argsort(prob_flat)[:, -self.beam_size:]
            y_mod = T.mod(y, self.vocab_size)
            y_mod = y_mod * y_mm
            y_mm = T.switch(
                T.eq(y_mod, 0),
                0,
                1)
            p_y_flat = prob_flat[y.flatten()] * y_mm
            a = y_mod.shape[0]
            b = y_mod.shape[1]
            y_emb = T.reshape(wemb[y_mod.flatten()], [a, b, self.emb_dim])
            y_emb = y_emb.dimshuffle(1, 0, 2)
            return [y_emb, s, c, y_mm, p_y_flat], theano.scan_module.until(T.all(T.eq(y_mm, 0)))

        def step_gru(y_emb_, s_, y_mm_, is_first, pctx, ctx, x_m, swt, sbi, swtg, sbig, su, sug, adwt, acwt, acbi, gwt,
                     gbi, gwtg,
                     gbig, gu, gug, erwt, erbi, epwt, egwt, edwt, wemb):

            yi = T.dot(y_emb_, swt) + sbi
            ygi = T.dot(y_emb_, swtg) + sbig
            s, ci = self.feedstep()(yi, ygi, s_, pctx, ctx, x_m, su, sug, adwt, acwt, acbi, gwt, gbi, gwtg, gbig, gu,
                                    gug)
            prob = self.children['emitter'].feedforward([s, y_emb_, ci], {
                'emitter': {'recurent': {'Wt': erwt, 'Bi': erbi}, 'peek': {'Wt': epwt}, 'glimpse': {'Wt': egwt},
                            'predict': {'Wt': edwt}}})
            prob = prob * y_mm_.flatten()[:,None]
            prob_flat = ifelse(T.eq(is_first, 1), prob[::self.beam_size, :],
                               prob.reshape([self.n_samples,self.beam_size*self.vocab_size]))
            y_flat = T.argsort(prob_flat)[:, -self.beam_size:]
            y_mod = T.mod(y_flat, self.vocab_size)
            y_emb = T.reshape(wemb[T.cast(y_mod.flatten(), 'int64')], [self.n_samples, self.beam_size, self.emb_dim])
            y_mm_c = T.switch(T.eq(y_mod, 0), 0., 1.)
            n_zero=T.cast(self.beam_size-y_mm_.sum(-1),'int32')
            y_mm_p=T.arange(0,self.beam_size)[None,:]
            y_mm_shifted=T.switch(y_mm_p<n_zero[:,None],0,1)
            p_y_flat=-T.log(prob_flat[T.repeat(T.arange(self.n_samples),self.beam_size),y_flat.flatten()].reshape([self.n_samples,self.beam_size]))*y_mm_shifted
            y_mm=y_mm_c*y_mm_shifted
            return [y_emb, s, y_mm, T.constant(0),y_mm_shifted , y_flat,y_mod, p_y_flat], theano.scan_module.until(T.eq(y_mm.sum(),0))

        step = None
        if self.core == lstm:
            step = step_lstm
        elif self.core == gru:
            step = step_gru

        result, updates = theano.scan(step, sequences=[], outputs_info=initiate_state,
                                      non_sequences=context, strict=True, n_steps=50)
        self.gen_updates = updates

        if self.core == lstm:
            y_emb, s, c, self.y_mm, is_first,self.y_mm_shifted, self.y_flat,self.y_mod, self.p_y_flat = result
        elif self.core == gru:
            y_emb, s, self.y_mm, is_first,self.y_mm_shifted, self.y_flat,self.y_mod, self.p_y_flat  = result

        def dec(mask, y_flat, y_mod, prob, y_idx_,y_prob_):
            ax0=T.repeat(T.arange(self.n_samples),self.beam_size)

            mask=mask[ax0,y_idx_.flatten()].reshape([self.n_samples,self.beam_size])
            y_flat=y_flat[ax0,y_idx_.flatten()].reshape([self.n_samples,self.beam_size])
            y_mod=y_mod[ax0,y_idx_.flatten()].reshape([self.n_samples,self.beam_size])
            prob=prob[ax0,y_idx_.flatten()].reshape([self.n_samples,self.beam_size])

            y_prob=y_prob_+prob
            y_before= y_flat // self.vocab_size
            y_out=T.switch(T.eq(mask, 1), y_mod, 0)
            y_idx=T.switch(T.eq(mask, 1), y_before, T.tile(T.arange(self.beam_size),self.n_samples).reshape([self.n_samples,self.beam_size]))

            return y_idx,y_prob,y_out,mask

        i_y_idx=T.tile(T.arange(self.beam_size), self.n_samples).reshape([self.n_samples, self.beam_size])
        i_y_prob=T.zeros([self.n_samples, self.beam_size])
        i_y_prob = T.unbroadcast(i_y_prob, 1)
        [y_idx,y_prob,y_out,mask], updates = theano.scan(dec, sequences=[self.y_mm_shifted,self.y_flat,self.y_mod,self.p_y_flat], outputs_info=[i_y_idx,i_y_prob,None,None],go_backwards=True)
        self.sample_updates=OrderedDict()
        self.sample_updates.update(self.gen_updates)
        self.sample_updates.update(self.updates)
        idx=(y_prob[-1]/mask.sum(0)).argmin(-1)
        self.sample=y_out.dimshuffle(1,2,0)[:,:,::-1][T.arange(self.n_samples),idx]


    def get_cost(self, Y):
        cost = T.nnet.categorical_crossentropy(self.output_cost, Y.flatten())
        cost = cost.reshape([Y.shape[0], Y.shape[1]])
        cost = (cost * self.y_mask).sum(0)
        self.cost = T.mean(cost)

    def get_error(self, Y):
        self.error = T.sum(T.neq(Y, self.predict) * self.y_mask) / T.sum(self.y_mask)
