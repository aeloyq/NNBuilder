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

    def get_output(self):
        self.get_n_samples()
        self.prepare()
        output = self.apply()
        self.output, updates = self.scan(output[0], output[1])
        self.get_context()
        self.updates.update(updates)

    def add_mask(self, tvar, mask):
        return mask[:, None] * tvar + (1. - mask)[:, None] * tvar

    def apply(self):
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
                               non_sequences=self.context, go_backwards=self.go_backwards)
        else:
            return theano.scan(step_mask, sequences=self.state_before, outputs_info=self.initiate_state,
                               non_sequences=self.context, n_steps=self.n_steps, go_backwards=self.go_backwards)

    def feedstep(self):
        self.merge()
        return self.apply()

    def feedscan(self, X=None):
        self.init_children()
        if X is None:
            X = self.input
        self.set_input(X)
        self.get_output()
        self.merge()
        return self.output

    def slice(self, _x, n, dim):
        return _x[:, n * dim:(n + 1) * dim]


class rnn(sequential):
    def __init__(self, unit, mask=True, activation=T.tanh, **kwargs):
        sequential.__init__(self, mask=mask, **kwargs)
        self.mask = mask
        self.unit_dim = unit
        self.activation = activation
        self.children['linear_bias'] = linear_bias(unit)

    def init_params(self):
        self.u = self.allocate(orthogonal, 'U', weight, self.unit_dim, self.unit_dim)

    def prepare(self):
        input = self.children['lb'].feedforward(self.input)
        self.state_before = [input]
        if self.mask:
            self.state_before.append(self.x_mask)
        self.initiate_state = [T.zeros([self.n_samples, self.unit_dim], theano.config.floatX)]
        self.context = None
        self.n_steps = input.shape[0]

    def apply(self):
        def step(x, h_):
            h = T.dot(h_, self.u) + x
            if self.activation is not None:
                h = self.activation(h)
            self.addops('hidden_unit', h, dropout)
            return h

        def step_mask(x, m, h_):
            h = T.dot(h_, self.u) + x
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
        self.children['input'] = linear_bias(unit)
        self.children['gate'] = linear_bias(unit * 2)
        self.condition = None
        self.setattr('out')
        self.setattr('condition')

    def init_params(self):
        self.u = self.allocate(orthogonal, 'U', weight, self.unit_dim, self.unit_dim)
        self.ug = self.allocate(orthogonal, 'Ug', weight, self.unit_dim, self.unit_dim * 2)

    def prepare(self):
        self.state_before = [self.children['input'].feedforward(self.input),
                             self.children['gate'].feedforward(self.input)]
        if self.mask:
            self.state_before.append(self.x_mask)
        self.initiate_state = [T.zeros([self.n_samples, self.unit_dim], theano.config.floatX)]
        self.context = None
        self.n_steps = self.input.shape[0]

    def apply(self):
        def step(x, xg, h_):
            gate = xg + T.dot(h_, self.ug)
            gate = self.conditional(gate)
            gate = T.nnet.sigmoid(gate)
            z_gate = self.slice(gate, 0, self.unit_dim)
            self.addops('z_gate', z_gate, dropout)
            r_gate = self.slice(gate, 1, self.unit_dim)
            self.addops('r_gate', r_gate, dropout)
            h_c = T.tanh(x + T.dot(r_gate * h_, self.u))
            h = (1 - z_gate) * h_ + z_gate * h_c
            self.addops('hidden_unit', h, dropout, False)
            return h

        def step_mask(x, xg, m, h_):
            gate = xg + T.dot(h_, self.ug)
            gate = self.conditional(gate)
            gate = T.nnet.sigmoid(gate)
            z_gate = self.slice(gate, 0, self.unit_dim)
            self.addops('z_gate', z_gate, dropout)
            r_gate = self.slice(gate, 1, self.unit_dim)
            self.addops('r_gate', r_gate, dropout)
            h_c = T.tanh(x + T.dot(r_gate * h_, self.u))
            h = (1 - z_gate) * h_ + z_gate * h_c
            h = self.add_mask(h, m)
            self.addops('hidden_unit', h, dropout, False)
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
        self.children['input'] = linear_bias(unit * 4)
        self.condition = None
        self.setattr('out')
        self.setattr('condition')

    def init_params(self):
        self.u = self.allocate(orthogonal, 'U', weight, self.unit_dim, self.unit_dim * 4)

    def prepare(self):
        self.state_before = [self.children['input'].feedforward(self.input)]
        if self.mask:
            self.state_before.append(self.x_mask)
        self.initiate_state = [T.zeros([self.n_samples, self.unit_dim], theano.config.floatX),
                               T.zeros([self.n_samples, self.unit_dim], theano.config.floatX)]
        self.context = None
        self.n_steps = self.input.shape[0]

    def get_output(self):
        self.get_n_samples()
        self.prepare()
        output = self.apply()
        self.output, updates = self.scan(output[0], output[1])
        self.output = self.output[0]
        self.get_context()
        self.updates.update(updates)

    def apply(self):
        def step(x, h_, c_):
            gate = x + T.dot(h_, self.u)
            gate = self.conditional(gate, h_)
            gate = T.nnet.sigmoid(gate)
            f_gate = self.slice(gate, 0, self.unit_dim)
            self.addops('f_gate', f_gate, dropout)
            i_gate = self.slice(gate, 1, self.unit_dim)
            self.addops('i_gate', i_gate, dropout)
            o_gate = self.slice(gate, 2, self.unit_dim)
            self.addops('o_gate', o_gate, dropout)
            cell = self.slice(gate, 3, self.unit_dim)
            c = f_gate * c_ + i_gate * cell
            self.addops('cell_unit', c, dropout, False)
            h = o_gate * T.nnet.sigmoid(c)
            self.addops('hidden_unit', h, dropout, False)
            return h, c

        def step_mask(x, m, h_, c_):
            gate = x + T.dot(h_, self.u)
            gate = self.conditional(gate, h_)
            gate = T.nnet.sigmoid(gate)
            f_gate = self.slice(gate, 0, self.unit_dim)
            self.addops('f_gate', f_gate, dropout)
            i_gate = self.slice(gate, 1, self.unit_dim)
            self.addops('i_gate', i_gate, dropout)
            o_gate = self.slice(gate, 2, self.unit_dim)
            self.addops('o_gate', o_gate, dropout)
            cell = self.slice(gate, 3, self.unit_dim)
            c = f_gate * c_ + i_gate * cell
            c = self.add_mask(c, m)
            self.addops('cell_unit', c, dropout, False)
            h = o_gate * T.nnet.sigmoid(c)
            h = self.add_mask(h, m)
            self.addops('hidden_unit', h, dropout, False)
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
        if self.structure == 'single':
            self.children['forward'] = core(unit, self.mask)
        elif self.structure == 'bi':
            self.children['forward'] = core(unit, self.mask)
            self.children['backward'] = core(unit, self.mask, go_backwards=True)

    def get_output(self):
        self.output = self.apply()

    def apply(self):
        if self.structure == 'single':
            return self.children['forward'].feedscan(self.input)
        elif self.structure == 'bi':
            fwd = self.children['forward'].feedscan(self.input)
            bwd = self.children['backward'].feedscan(self.input)
            return concatenate([fwd, bwd[::-1]], 2)


class attention(layer):
    def __init__(self, unit, h_dim, s_dim, o_dim=1, **kwargs):
        layer.__init__(self, unit, **kwargs)
        self.h_dim = h_dim
        self.s_dim = s_dim
        self.o_dim = o_dim
        self.y = None
        self.children['enc_h'] = linear_bias(self.unit_dim, in_dim=self.h_dim)
        self.children['dec_s'] = linear(self.unit_dim, in_dim=self.s_dim)
        self.children['combine'] = linear_bias(self.o_dim, in_dim=self.unit_dim)

    def apply(self):
        s_ = self.input[0]
        ctx = self.input[1]
        mask = self.input[2]
        att_layer_1 = self.children['enc_h'].feedforward(ctx) + self.children['dec_s'].feedforward(s_)
        att_layer_2 = self.children['combine'].feedforward(att_layer_1)
        if self.o_dim == 1:
            att_layer_2 = att_layer_2.reshape([att_layer_2.shape[0], att_layer_2.shape[1]])
        eij = T.exp(att_layer_2) * mask
        aij = eij / eij.sum(0, keepdims=True)
        return aij


class emitter(layer):
    def __init__(self, unit, emb_dim=0, structure='direct', gen='train', **kwargs):
        layer.__init__(self, unit, **kwargs)
        self.emb_dim = emb_dim
        self.structure = structure
        self.gen = gen
        if self.structure == 'direct':
            self.children['emitter'] = std(self.unit_dim)
        elif self.structure == 'hierarchical':
            self.children['readout'] = readout(self.emb_dim)
            self.children['lookup'] = std(self.unit_dim, in_dim=self.emb_dim)

    def apply(self):
        emit_word = None
        emitted_word = None
        if self.structure == 'direct':
            emit_word = T.nnet.softmax(self.children['emitter'].feedforward(self.input))
        elif self.structure == 'hierarchical':
            target_embedding = self.children['readout'].feedforward()
            emit_word = T.nnet.softmax(self.children['lookup'].feedforward(target_embedding))
        if self.gen == 'train':
            emitted_word = emit_word.argmax(axis=-1)
        elif self.gen == 'stochastic':
            emitted_word = self.trng(pvals=emit_word).argmax(axis=-1)
        elif self.gen == 'beamsearch':
            emitted_word = self.trng(pvals=emit_word).argmax(axis=-1)
        return emit_word, emitted_word


class decoder(sequential):
    def __init__(self, unit, emb_dim, vocab_size, core=gru, ctx='attention_bi', structure='glimpse',
                 state_initiate='mean', emitter_strcuture='direct', is_maxout=True, feedback=True, **kwargs):
        sequential.__init__(self, **kwargs)
        self.unit_dim = unit
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.core = core
        self.ctx = ctx
        self.structure = structure
        self.state_initiate = state_initiate
        self.emitter_strcuture = emitter_strcuture
        self.attention_unit = unit
        self.maxout = is_maxout
        self.feedback = feedback
        self.mask = True
        self.setattr('attention_unit')
        self.y = None
        self.setattr('in_dim')
        self.set_children()

    def set_children(self):
        if self.structure == 'naive':
            self.children['state_dec'] = self.core(self.unit_dim, mask=False)
        elif self.structure == 'glimpse':
            self.children['state_dec'] = self.core(self.unit_dim, mask=False, in_dim=self.emb_dim)
            if self.ctx == 'attention_bi':
                self.children['glimpse_dec'] = self.core(self.unit_dim, mask=False, in_dim=self.in_dim * 2)
            else:
                self.children['glimpse_dec'] = self.core(self.unit_dim, mask=False, in_dim=self.in_dim)
        dim_before_emit = self.unit_dim
        if self.maxout:
            self.children['maxout'] = maxout(2)
            dim_before_emit = dim_before_emit / 2
        self.children['emitter_train'] = emitter(self.vocab_size, self.emb_dim, self.emitter_strcuture,
                                                 in_dim=dim_before_emit)
        if self.feedback:
            self.children['peek'] = lookuptable(self.emb_dim, in_dim=self.vocab_size)

        if self.ctx == 'attention':
            self.children['attention'] = attention(self.attention_unit, self.in_dim, self.unit_dim)
        elif self.ctx == 'attention_bi':
            self.children['attention'] = attention(self.attention_unit, self.in_dim * 2, self.unit_dim)

    def prepare(self):
        if self.feedback:
            self.state_before = self.children['peek'].feedforward(self.y)
            emb_shifted = T.zeros_like(self.state_before)
            emb_shifted = T.set_subtensor(emb_shifted[1:], self.state_before[:-1])
            first_unit = self.children['state_dec']
            if self.core == lstm:
                y = first_unit.children['input'].feedforward(emb_shifted)
                self.state_before = [y, self.y, self.y_mask]
            elif self.core == gru:
                y = first_unit.children['input'].feedforward(emb_shifted)
                yg = first_unit.children['gate'].feedforward(emb_shifted)
                self.state_before = [y, yg, self.y, self.y_mask]
        else:
            self.state_before = [self.y, self.y_mask]
        mean_ctx = (self.input[:, :, -self.in_dim:] * self.x_mask[:, :, None]).sum(0) / self.x_mask.sum(0)[:, None]
        s_0 = mean_ctx
        if self.state_initiate == 'final':
            s_0 = self.input[0, :, -self.in_dim:]
        self.initiate_state = [s_0]
        if isinstance(self.core, lstm):
            self.initiate_state.append(T.zeros([self.n_samples, self.unit_dim], theano.config.floatX))
        self.initiate_state.append(None)
        self.initiate_state.append(None)
        if self.ctx == 'final':
            self.children['glimpse_dec'] = linear(self.unit_dim, self.in_dim)
            self.context = [self.children['glimpse_dec'].feedforward(self.input[-1][:, -self.in_dim:])]
        elif self.ctx == 'mean':
            self.children['glimpse_dec'] = linear(self.unit_dim, self.in_dim)
            self.context = [self.children['glimpse_dec'].feedforward(mean_ctx)]
        elif self.ctx == 'attention':
            self.context = [self.input]
        elif self.ctx == 'attention_bi':
            self.context = [self.input]
        self.context.append(self.x_mask)
        self.n_steps = self.y.shape[0]

    def get_output(self):
        self.get_n_samples()
        self.prepare()
        output = self.apply()
        self.output, updates = self.scan(output[0], output[1])
        self.output = self.output[-2]
        self.cost = self.output[-1].mean()
        self.updates.update(updates)

    def apply(self):
        def step_lstm(y, y_t, y_m, s_, c_, ctx, x_m):
            if self.ctx == 'attention' or self.ctx == 'attention_bi':
                aij = self.children['attention'].feedforward([s_, ctx, x_m])

            def condition():
                if self.ctx == 'final' or self.ctx == 'mean':
                    return ctx
                elif self.ctx == 'attention' or self.ctx == 'attention_bi':
                    ci = (ctx * aij[:, :, None]).sum(0)
                    return ci, aij

            s = None
            c = None
            if self.structure == 'naive':
                self.children['state_dec'].condition = condition
                s, c = self.children['state_dec'].feedstep()[0](y, s_, c_)
            elif self.structure == 'glimpse':
                s1, c1 = self.children['state_dec'].feedstep()[0](y, s_, c_)
                ci, aij = condition()
                condition = self.children['glimpse_dec'].children['input'].feedforward(ci)
                s2, c2 = self.children['glimpse_dec'].feedstep()[0](condition, s1, c1)
                s = s2
                c = c2
            if self.maxout:
                s = self.children['maxout'].feedforward(s)
            o, w = self.children['emitter_train'].feedforward(s)
            cost = T.nnet.categorical_crossentropy(o, y_t) * y_m
            return s, c, w, cost

        def step_gru(y, yg, y_t, y_m, s_, ctx, x_m):
            if self.ctx == 'attention' or self.ctx == 'attention_bi':
                aij = self.children['attention'].feedforward([s_, ctx, x_m])

            def condition():
                if self.ctx == 'final' or self.ctx == 'mean':
                    return ctx
                elif self.ctx == 'attention' or self.ctx == 'attention_bi':
                    ci = (ctx * aij[:, :, None]).sum(0)
                    return ci, aij

            s = None
            if self.structure == 'naive':
                self.children['state_dec'].condition = condition
                s = self.children['state_dec'].feedstep()[0](y, yg, s_)
            elif self.structure == 'glimpse':
                s1 = self.children['state_dec'].feedstep()[0](y, yg, s_)
                ci, aij = condition()
                condition = self.children['glimpse_dec'].children['input'].feedforward(ci)
                conditiong = self.children['glimpse_dec'].children['gate'].feedforward(ci)
                s2 = self.children['glimpse_dec'].feedstep()[0](condition, conditiong, s1)
                s = s2
            if self.maxout:
                s = self.children['maxout'].feedforward(s)
            o, w = self.children['emitter_train'].feedforward(s)
            cost = T.nnet.categorical_crossentropy(o, y_t) * y_m
            return s, w, cost

        return step_lstm, step_gru

    def scan(self, step_lstm, step_gru):
        if self.core == lstm:
            return theano.scan(step_lstm, sequences=self.state_before, outputs_info=self.initiate_state,
                               non_sequences=self.context, go_backwards=self.go_backwards)
        elif self.core == gru:
            return theano.scan(step_gru, sequences=self.state_before, outputs_info=self.initiate_state,
                               non_sequences=self.context, n_steps=self.n_steps, go_backwards=self.go_backwards)

    def get_predict(self):
        self.predict = self.output

    def get_cost(self, Y):
        self.cost = self.cost

    def get_error(self, Y):
        self.error = T.mean(T.neq(Y, self.predict))
