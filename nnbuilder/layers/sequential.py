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
        if self.input.ndim == 3:
            n_samples = self.input.shape[1]
        else:
            n_samples = 1
        self.n_samples = n_samples

    def prepare(self):
        self.state_before = [self.input]
        if self.mask: self.state_before = [self.input, self.x_mask]
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
        self.prepare()
        output = self.apply(X, P)
        self.output, updates = self.scan(output[0], output[1])
        self.get_context()
        self.updates.update(updates)

    def add_mask(self, tvar, mask):
        return mask[:, None] * tvar + (1. - mask)[:, None] * tvar

    def apply(self, X, P):
        def step(x):
            h = x
            self.addops('hidden_unit', h, dropout)
            return h

        def step_mask(x, m):
            h = x
            h = self.add_mask(h, m)
            self.addops('hidden_unit', h, dropout)
            return h

        return step, step_mask

    def scan(self, step, step_mask):
        if not self.mask:
            return theano.scan(step, sequences=self.state_before, outputs_info=self.initiate_state,
                               non_sequences=self.context, n_steps=self.n_steps, go_backwards=self.go_backwards,
                               strict=True)
        else:
            return theano.scan(step_mask, sequences=self.state_before, outputs_info=self.initiate_state,
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
        return _x[:, n * dim:(n + 1) * dim]


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

    def prepare(self):
        input = self.children['lb'].feedforward(self.input)
        self.state_before = [input]
        if self.mask:
            self.state_before.append(self.x_mask)
        self.initiate_state = [T.zeros([self.n_samples, self.unit_dim], theano.config.floatX)]
        self.context = [self.u]
        self.n_steps = input.shape[0]

    def apply(self, X, P):
        def step(x, h_, u):
            h = T.dot(h_, u) + x
            if self.activation is not None:
                h = self.activation(h)
            self.addops('hidden_unit', h, dropout)
            return h

        def step_mask(x, m, h_, u):
            h = T.dot(h_, u) + x
            if self.activation is not None:
                h = self.activation(h)
            h = self.add_mask(h, m)
            self.addops('hidden_unit', h, dropout)
            return h

        return step, step_mask


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

    def prepare(self):
        self.state_before = [self.children['input'].feedforward(self.input),
                             self.children['gate'].feedforward(self.input)]
        if self.mask:
            self.state_before.append(self.x_mask)
        self.initiate_state = [T.zeros([self.n_samples, self.unit_dim], theano.config.floatX)]
        self.context = [self.u, self.ug]
        self.n_steps = self.input.shape[0]

    def apply(self, X, P):
        def step(x, xg, h_, u, ug):
            gate = xg + T.dot(h_, ug)
            gate = self.conditional(gate)
            gate = T.nnet.sigmoid(gate)
            z_gate = self.slice(gate, 0, self.unit_dim)
            self.addops('z_gate', z_gate, dropout, False)
            r_gate = self.slice(gate, 1, self.unit_dim)
            self.addops('r_gate', r_gate, dropout, False)
            h_c = T.tanh(x + T.dot(r_gate * h_, u))
            h = (1 - z_gate) * h_ + z_gate * h_c
            self.addops('hidden_unit', h, dropout)
            return h

        def step_mask(x, xg, m, h_, u, ug):
            gate = xg + T.dot(h_, ug)
            gate = self.conditional(gate)
            gate = T.nnet.sigmoid(gate)
            z_gate = self.slice(gate, 0, self.unit_dim)
            self.addops('z_gate', z_gate, dropout, False)
            r_gate = self.slice(gate, 1, self.unit_dim)
            self.addops('r_gate', r_gate, dropout, False)
            h_c = T.tanh(x + T.dot(r_gate * h_, u))
            h = (1 - z_gate) * h_ + z_gate * h_c
            h = self.add_mask(h, m)
            self.addops('hidden_unit', h, dropout)
            return h

        return step, step_mask

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

    def prepare(self):
        self.state_before = [self.children['input'].feedforward(self.input)]
        if self.mask:
            self.state_before.append(self.x_mask)
        self.initiate_state = [T.zeros([self.n_samples, self.unit_dim], theano.config.floatX),
                               T.zeros([self.n_samples, self.unit_dim], theano.config.floatX)]
        self.context = [self.u]
        self.n_steps = self.input.shape[0]

    def get_output(self, X, P):
        self.get_n_samples()
        self.prepare()
        output = self.apply(X, P)
        self.output, updates = self.scan(output[0], output[1])
        self.output = self.output[0]
        self.get_context()
        self.updates.update(updates)

    def apply(self, X, P):
        def step(x, h_, c_, u):
            gate = x + T.dot(h_, u)
            gate = self.conditional(gate, h_)
            gate = T.nnet.sigmoid(gate)
            f_gate = self.slice(gate, 0, self.unit_dim)
            self.addops('f_gate', f_gate, dropout, False)
            i_gate = self.slice(gate, 1, self.unit_dim)
            self.addops('i_gate', i_gate, dropout, False)
            o_gate = self.slice(gate, 2, self.unit_dim)
            self.addops('o_gate', o_gate, dropout, False)
            cell = self.slice(gate, 3, self.unit_dim)
            c = f_gate * c_ + i_gate * cell
            self.addops('cell_unit', c, dropout)
            h = o_gate * T.nnet.sigmoid(c)
            self.addops('hidden_unit', h, dropout)
            return h, c

        def step_mask(x, m, h_, c_, u):
            gate = x + T.dot(h_, u)
            gate = self.conditional(gate, h_)
            gate = T.nnet.sigmoid(gate)
            f_gate = self.slice(gate, 0, self.unit_dim)
            self.addops('f_gate', f_gate, dropout, False)
            i_gate = self.slice(gate, 1, self.unit_dim)
            self.addops('i_gate', i_gate, dropout, False)
            o_gate = self.slice(gate, 2, self.unit_dim)
            self.addops('o_gate', o_gate, dropout, False)
            cell = self.slice(gate, 3, self.unit_dim)
            c = f_gate * c_ + i_gate * cell
            c = self.add_mask(c, m)
            self.addops('cell_unit', c, dropout)
            h = o_gate * T.nnet.sigmoid(c)
            h = self.add_mask(h, m)
            self.addops('hidden_unit', h, dropout)
            return h, c

        return step, step_mask

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
            return concatenate([fwd, bwd[::-1]], 2)


class attention(layer):
    def __init__(self, unit, s_dim, o_dim=1, **kwargs):
        layer.__init__(self, unit, **kwargs)
        self.s_dim = s_dim
        self.o_dim = o_dim
        self.y = None
        self.children['dec_s'] = linear(self.unit_dim, in_dim=self.s_dim)
        self.children['combine'] = linear_bias(self.o_dim, in_dim=self.unit_dim)

    def apply(self, X, P):
        s_ = self.input[0]
        ctx = self.input[1]
        mask = self.input[2]
        att_layer_1 = T.tanh(ctx + self.children['dec_s'].feedforward(s_))
        self.addops('att_in', att_layer_1, dropout)
        att_layer_2 = self.children['combine'].feedforward(att_layer_1)
        if self.o_dim == 1:
            att_layer_2 = att_layer_2.reshape([att_layer_2.shape[0], att_layer_2.shape[1]])
        eij = T.exp(att_layer_2) * mask
        eij = eij - eij.max(0, keepdims=True)
        aij = eij / eij.sum(0, keepdims=True)
        return aij


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
        self.children['predict'] = softmax(self.unit_dim, in_dim=in_dim)

    def apply(self, X, P):
        s = self.input[0]
        y_ = self.input[1]
        ctx = self.input[2]
        prob = self.children['recurent'].feedforward(s) + self.children['peek'].feedforward(y_) + self.children[
            'glimpse'].feedforward(ctx)
        self.addops('emit_gate', prob, dropout)
        if self.maxout:
            prob = self.children['maxout'].feedforward(prob, P)
        else:
            prob = T.tanh(prob)
        emit_word = self.children['predict'].feedforward(prob, P)

        return emit_word


class decoder(sequential):
    def __init__(self, unit, emb_dim, vocab_size, core=gru,
                 state_initiate='mean', is_maxout=True, **kwargs):
        sequential.__init__(self, **kwargs)
        self.unit_dim = unit
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.core = core
        self.state_initiate = state_initiate
        self.attention_unit = unit
        self.maxout = is_maxout
        self.y = None
        self.beam_size = 12
        self.setattr('beam_size')
        self.setattr('attention_unit')

    def set_children(self):
        self.children['state_dec'] = self.core(self.unit_dim, mask=False, in_dim=self.emb_dim)
        self.children['glimpse_dec'] = self.core(self.unit_dim, mask=False, in_dim=self.in_dim * 2)
        self.children['emitter'] = emitter(self.vocab_size, self.in_dim * 2, self.unit_dim, self.emb_dim,
                                           in_dim=self.unit_dim, is_maxout=self.maxout)
        self.children['context'] = linear_bias(self.attention_unit, in_dim=self.in_dim * 2)
        self.children['attention'] = attention(self.attention_unit, self.unit_dim)
        self.children['peek'] = lookuptable(self.emb_dim, in_dim=self.vocab_size)

    def init_params(self):
        self.wt_iniate_s = self.allocate(uniform, 'Wt_iniate_s', weight, self.in_dim, self.unit_dim)

    def prepare(self):
        self.state_before = self.children['peek'].feedforward(self.y)
        emb_shifted = T.zeros_like(self.state_before)
        emb_shifted = T.set_subtensor(emb_shifted[1:], self.state_before[:-1])
        self.y_ = emb_shifted
        if self.core == lstm:
            y_ = self.children['state_dec'].children['input'].feedforward(emb_shifted)
            self.state_before = [y_]
        elif self.core == gru:
            y_ = self.children['state_dec'].children['input'].feedforward(emb_shifted)
            yg_ = self.children['state_dec'].children['gate'].feedforward(emb_shifted)
            self.state_before = [y_, yg_]

        mean_ctx = (self.input[:, :, -self.in_dim:] * self.x_mask[:, :, None]).sum(0) / self.x_mask.sum(0)[:, None]
        s_0 = mean_ctx
        if self.state_initiate == 'final': s_0 = self.input[0, :, -self.in_dim:]
        s_0 = T.dot(s_0, self.wt_iniate_s)
        self.initiate_state = [s_0]
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
        self.prepare()
        output = self.apply(X, P)
        [self.s, self.c], updates = self.scan(output[0], output[1])
        o = self.children['emitter'].feedforward([self.s, self.y_, self.c])
        self.output = o
        self.updates.update(updates)

    def apply(self, X, P):
        def step_lstm(y_, s_, c_, pctx, ctx, x_m, su, adwt, acwt, acbi, gwt, gbi, gu):
            s1, c1 = self.children['state_dec'].feedstep()[0](y_, s_, c_, su)
            aij = self.children['attention'].feedforward([s1, pctx, x_m],
                                                         {'dec_s': {'Wt': adwt}, 'combine': {'Wt': acwt, 'Bi': acbi}})
            ci = (ctx * aij[:, :, None]).sum(0)
            condition = T.dot(gwt, ci) + gbi
            s2, c2 = self.children['glimpse_dec'].feedstep()[0](condition, s1, c1, gu)
            s = s2
            return s, ci

        def step_gru(y_, yg_, s_, pctx, ctx, x_m, su, sug, adwt, acwt, acbi, gwt, gbi, gwtg, gbig, gu, gug):
            s1 = self.children['state_dec'].feedstep()[0](y_, yg_, s_, su, sug)
            aij = self.children['attention'].feedforward([s1, pctx, x_m], {
                'attention': {'dec_s': {'Wt': adwt}, 'combine': {'Wt': acwt, 'Bi': acbi}}})
            ci = (ctx * aij[:, :, None]).sum(0)
            condition = T.dot(ci, gwt) + gbi
            conditiong = T.dot(ci, gwtg) + gbig
            s2 = self.children['glimpse_dec'].feedstep()[0](condition, conditiong, s1, gu, gug)
            s = s2
            return s, ci

        return step_lstm, step_gru

    def scan(self, step_lstm, step_gru):
        if self.core == lstm:
            return theano.scan(step_lstm, sequences=self.state_before, outputs_info=self.initiate_state,
                               non_sequences=self.context, n_steps=self.n_steps, go_backwards=self.go_backwards,
                               strict=True)
        elif self.core == gru:
            return theano.scan(step_gru, sequences=self.state_before, outputs_info=self.initiate_state,
                               non_sequences=self.context, n_steps=self.n_steps, go_backwards=self.go_backwards,
                               strict=True)


    def get_predict(self):
        y_0 = T.zeros([self.n_samples, self.emb_dim])
        s_0 = (self.input[:, :, -self.in_dim:] * self.x_mask[:, :, None]).sum(0) / self.x_mask.sum(0)[:, None]
        if self.state_initiate == 'final': s_0 = self.input[0, :, -self.in_dim:]
        s_0 = T.dot(s_0, self.wt_iniate_s)
        initiate_state = [y_0, s_0]
        if isinstance(self.core, lstm):
            initiate_state.append(T.zeros([self.beam_size, self.n_samples, self.unit_dim], theano.config.floatX))
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
                     'Wt_emitter_predict_lb', 'Bi_emitter_predict_lb', 'Wemb_peek']
        elif self.core == lstm:
            plist = ['Wt_state_dec_input', 'Bi_state_dec_input', 'U_state_dec', 'Wt_attention_dec_s',
                     'Wt_attention_combine', 'Bi_attention_combine',
                     'Wt_glimpse_dec_input', 'Bi_glimpse_dec_input', 'U_glimpse_dec',
                     'Wt_emitter_recurent', 'Bi_emitter_recurent', 'Wt_emitter_peek', 'Wt_emitter_glimpse',
                     'Wt_emitter_predict_lb', 'Bi_emitter_predict_lb', 'Wemb_peek']
        for i in plist:
            ii = i.split('_')
            iii = ii[0] + '_' + self.name
            for j in range(1, len(ii)):
                iii = iii + '_' + ii[j]
            context.append(self.params[iii])

        from theano.ifelse import ifelse

        def slice(_x, n, dim):
            return _x[ :, n * dim:(n + 1) * dim]

        def step_lstm(y_emb_, s_, c_, y_mm, pctx, ctx, x_m, swt, sbi, su, adwt, acwt, acbi, gwt, gbi, gu, erwt, erbi,
                      epwt, egwt, edwt, edbi, wemb):
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
            prob = T.dot(prob, edwt) + edbi
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

        def step_gru(y_emb_, s_, pctx, ctx, x_m, swt, sbi, swtg, sbig, su, sug, adwt, acwt, acbi, gwt,
                     gbi, gwtg,
                     gbig, gu, gug, erwt, erbi, epwt, egwt, edwt, edbi, wemb):
            yi = T.dot(y_emb_, swt) + sbi
            ygi = T.dot(y_emb_, swtg) + sbig

            gate = ygi + T.dot(s_, sug)
            gate = T.nnet.sigmoid(gate)
            z_gate = slice(gate, 0, self.unit_dim)
            self.addops('state_dec_z_gate', z_gate, dropout, False)
            r_gate = slice(gate, 1, self.unit_dim)
            self.addops('state_dec_r_gate', r_gate, dropout, False)
            s_c = T.tanh(yi + T.dot(r_gate * s_, su))
            s = (1 - z_gate) * s_ + z_gate * s_c
            self.addops('state_dec_hidden_unit', s, dropout)
            s1 = s

            ps = T.dot(s1, adwt)
            att_layer_1 = T.tanh(pctx + ps)
            self.addops('attention_att_in', att_layer_1, dropout)
            att_layer_2 = T.dot(att_layer_1, acwt) + acbi
            att_layer_2 = att_layer_2.reshape([att_layer_2.shape[0], att_layer_2.shape[1]])
            eij = T.exp(att_layer_2) * x_m
            eij = eij - eij.max(0, keepdims=True)
            aij = eij / eij.sum(0, keepdims=True)

            ci = (ctx* aij[:, :, None]).sum(0)
            condition = T.dot(ci, gwt) + gbi
            conditiong = T.dot(ci, gwtg) + gbig

            gate = conditiong + T.dot(s1, gug)
            gate = T.nnet.sigmoid(gate)
            z_gate = slice(gate, 0, self.unit_dim)
            self.addops('glimpse_dec_z_gate', z_gate, dropout, False)
            r_gate = slice(gate, 1, self.unit_dim)
            self.addops('glimpse_dec_r_gate', r_gate, dropout, False)
            s_c = T.tanh(condition + T.dot(r_gate * s1, gu))
            s = (1 - z_gate) * s1 + z_gate * s_c
            self.addops('glimpse_dec_hidden_unit', s, dropout)

            prob = T.dot(s, erwt) + T.dot(y_emb_, epwt) + T.dot(ci, egwt) + erbi
            self.addops('emitter_emit_gate', prob, dropout)
            if self.maxout:
                prob = self.children['emitter'].children['maxout'].feedforward(prob)
            else:
                prob = T.tanh(prob)
            prob = T.nnet.softmax(T.dot(prob, edwt) + edbi)
            prob = self.trng.multinomial(pvals=prob)
            pred=prob.argmax(-1)
            y_emb = T.reshape(wemb[pred.flatten()], [self.n_samples,self.emb_dim])
            return [y_emb, s, pred], theano.scan_module.until(T.all(T.eq(pred, 0)))

        step = None
        if self.core == lstm:
            step = step_lstm
        elif self.core == gru:
            step = step_gru

        result, update = theano.scan(step, sequences=[], outputs_info=initiate_state,
                                     non_sequences=context, strict=True, n_steps=50)

        if self.core == lstm:
            y_emb, s, pred = result
        elif self.core == gru:
            y_emb, s, pred = result


        self.predict = pred

    def gen_sample(self):
        y_0 = T.zeros([self.beam_size, self.n_samples, self.emb_dim])
        y_mm_0 = T.ones([self.n_samples, self.beam_size])
        s_0 = (self.input[:, :, -self.in_dim:] * self.x_mask[:, :, None]).sum(0) / self.x_mask.sum(0)[:, None]
        if self.state_initiate == 'final': s_0 = self.input[0, :, -self.in_dim:]
        s_0 = T.dot(s_0, self.wt_iniate_s)
        s_0 = (T.tile(s_0, self.beam_size)).reshape([self.beam_size, self.n_samples, self.unit_dim])
        initiate_state = [y_0, s_0]
        if isinstance(self.core, lstm):
            initiate_state.append(T.zeros([self.beam_size, self.n_samples, self.unit_dim], theano.config.floatX))
        initiate_state.append(y_mm_0)
        initiate_state.append(T.constant(1))
        initiate_state.append(None)
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
                     'Wt_emitter_predict_lb', 'Bi_emitter_predict_lb', 'Wemb_peek']
        elif self.core == lstm:
            plist = ['Wt_state_dec_input', 'Bi_state_dec_input', 'U_state_dec', 'Wt_attention_dec_s',
                     'Wt_attention_combine', 'Bi_attention_combine',
                     'Wt_glimpse_dec_input', 'Bi_glimpse_dec_input', 'U_glimpse_dec',
                     'Wt_emitter_recurent', 'Bi_emitter_recurent', 'Wt_emitter_peek', 'Wt_emitter_glimpse',
                     'Wt_emitter_predict_lb', 'Bi_emitter_predict_lb', 'Wemb_peek']
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
                      epwt, egwt, edwt, edbi, wemb):
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
            prob = T.dot(prob, edwt) + edbi
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

        def step_gru(y_emb_, s_, y_mm, is_first, pctx, ctx, x_m, swt, sbi, swtg, sbig, su, sug, adwt, acwt, acbi, gwt,
                     gbi, gwtg,
                     gbig, gu, gug, erwt, erbi, epwt, egwt, edwt, edbi, wemb):
            yi = T.dot(y_emb_, swt) + sbi
            ygi = T.dot(y_emb_, swtg) + sbig

            gate = ygi + T.dot(s_, sug)
            gate = T.nnet.sigmoid(gate)
            z_gate = slice(gate, 0, self.unit_dim)
            self.addops('state_dec_z_gate', z_gate, dropout, False)
            r_gate = slice(gate, 1, self.unit_dim)
            self.addops('state_dec_r_gate', r_gate, dropout, False)
            s_c = T.tanh(yi + T.dot(r_gate * s_, su))
            s = (1 - z_gate) * s_ + z_gate * s_c
            self.addops('state_dec_hidden_unit', s, dropout)
            s1 = s

            ps = T.tile(T.dot(s1, adwt), pctx.shape[0]).reshape(
                [pctx.shape[0], self.beam_size, self.n_samples, self.attention_unit])
            ps = ps.dimshuffle(1, 0, 2, 3)
            att_layer_1 = T.tanh(pctx + ps)
            self.addops('attention_att_in', att_layer_1, dropout)
            att_layer_2 = T.dot(att_layer_1, acwt) + acbi
            att_layer_2 = att_layer_2.reshape([att_layer_2.shape[0], att_layer_2.shape[1], att_layer_2.shape[2]])
            eij = T.exp(att_layer_2) * x_m[None, :, :]
            eij = eij - eij.max(1, keepdims=True)
            aij = eij / eij.sum(1, keepdims=True)

            ci = (ctx[None, :, :, :] * aij[:, :, :, None]).sum(1)
            condition = T.dot(ci, gwt) + gbi
            conditiong = T.dot(ci, gwtg) + gbig

            gate = conditiong + T.dot(s1, gug)
            gate = T.nnet.sigmoid(gate)
            z_gate = slice(gate, 0, self.unit_dim)
            self.addops('glimpse_dec_z_gate', z_gate, dropout, False)
            r_gate = slice(gate, 1, self.unit_dim)
            self.addops('glimpse_dec_r_gate', r_gate, dropout, False)
            s_c = T.tanh(condition + T.dot(r_gate * s1, gu))
            s = (1 - z_gate) * s1 + z_gate * s_c
            self.addops('glimpse_dec_hidden_unit', s, dropout)

            prob = T.dot(s, erwt) + T.dot(y_emb_, epwt) + T.dot(ci, egwt) + erbi
            self.addops('emitter_emit_gate', prob, dropout)
            if self.maxout:
                prob = self.children['emitter'].children['maxout'].feedforward(prob)
            else:
                prob = T.tanh(prob)
            prob = T.nnet.softmax(T.dot(prob, edwt) + edbi)
            prob = prob.dimshuffle(1, 0, 2)
            prob = prob * y_mm[:, :, None]
            # prob_flat =prob.reshape([prob.shape[0], prob.shape[1] * prob.shape[2]])
            prob_flat = ifelse(T.eq(is_first, 1), T.reshape(prob[:, 0, :], [prob.shape[0], prob.shape[2]]),
                               prob.reshape([prob.shape[0], prob.shape[1] * prob.shape[2]]))
            y_flat = T.argsort(prob_flat)[:, -self.beam_size:]
            y_mod = T.mod(y_flat, self.vocab_size)
            y_mod = y_mod * y_mm
            y_mm = T.switch(T.eq(y_mod, 0), 0., 1.)
            p_y_flat = prob_flat[y_flat.flatten()] * y_mm
            a = y_mod.shape[0]
            b = y_mod.shape[1]
            y_emb = T.reshape(wemb[T.cast(y_mod.flatten(), 'int64')], [a, b, self.emb_dim])
            y_emb = y_emb.dimshuffle(1, 0, 2)
            return [y_emb, s, y_mm, T.constant(0), y_flat, p_y_flat], theano.scan_module.until(T.all(T.eq(y_mm, 0)))

        step = None
        if self.core == lstm:
            step = step_lstm
        elif self.core == gru:
            step = step_gru

        result, update = theano.scan(step, sequences=[], outputs_info=initiate_state,
                                     non_sequences=context, strict=True, n_steps=50)

        if self.core == lstm:
            y_emb, s, y_mm, is_first, y_flat, p_y_flat = result
        elif self.core == gru:
            y_emb, s, y_mm, is_first, y_flat, p_y_flat = result

        self.y_flat = y_flat
        self.p_y_flat=p_y_flat

    def get_cost(self, Y):
        y_flat = Y.flatten()
        y_flat_idx = T.arange(y_flat.shape[0]) * self.vocab_size + y_flat
        cost = -T.log((self.output.flatten() + 1e-8)[y_flat_idx])
        cost = cost.reshape([Y.shape[0], Y.shape[1]])
        self.cost = (cost * self.y_mask).sum(0)
        self.cost = T.mean(self.cost)

    def get_error(self, Y):
        self.error = T.mean(T.neq(Y, self.predict) * self.y_mask)
