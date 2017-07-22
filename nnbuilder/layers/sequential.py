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
        self.full = True
        self.other = []
        self.setattr('out')
        self.setattr('mask')
        self.setattr('y_mask')
        self.setattr('x_mask')
        self.setattr('go_backwards')

    def get_n_samples(self):
        self.n_samples = self.input.shape[-2]

    def prepare(self, X):
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

    def get_output(self, X):
        self.get_n_samples()
        self.prepare(X)
        output, updates = self.scan()
        if not self.full:
            self.output = output[0]
            self.other = output[1:]
        else:
            self.output = output
        self.get_context()
        self.updates.update(updates)

    def add_mask(self, tvar, args):
        '''
        mask here refers to mask in each step
        so its shape must be batchsize(n_samples)
        assert location of dim refers to batchsize should be -2
        :param tvar: tensor variable which need to be masked
        :param mask: sizeof batchsize
        :return:
        '''
        if not self.mask:
            return tvar
        mask = args[-1 - len(self.context)]
        if tvar.ndim == 2:
            return mask[:, None] * tvar + (1. - mask)[:, None] * tvar
        elif tvar.ndim == 3:
            return mask[None, :, None] * tvar + (1. - mask)[None, :, None] * tvar
        elif tvar.ndim == 4:
            return mask[None, None, :, None] * tvar + (1. - mask)[None, None, :, None] * tvar

    def step(self, *args):
        x = args[0]
        h = x
        h = self.add_mask(h, args)
        h = self.addops('hidden_unit', h, dropout)
        return h

    def scan(self):
        return theano.scan(self.step, sequences=self.state_before, outputs_info=self.initiate_state,
                           non_sequences=self.context, n_steps=self.n_steps, go_backwards=self.go_backwards)

    def feedscan(self, X=None, P=None):
        if X is None:
            X = self.input
        self.set_input(X)
        self.get_output(X)
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

    def prepare(self, X):
        input = self.children['lb'].feedforward(X)
        self.state_before = [input]
        if self.mask:
            self.state_before.append(self.x_mask)
        self.initiate_state = [T.zeros([self.n_samples, self.unit_dim], theano.config.floatX)]
        self.context = []
        self.n_steps = input.shape[0]

    def step(self, *args):
        x = args[0]
        h_ = args[1]
        h = T.dot(h_, self.u) + x
        if self.activation is not None:
            h = self.activation(h)
        h = self.add_mask(h, args)
        h = self.addops('hidden_unit', h, dropout)
        return h


class gru(sequential):
    def __init__(self, unit, mask=True, **kwargs):
        sequential.__init__(self, mask=mask, **kwargs)
        self.mask = mask
        self.out = 'all'
        self.unit_dim = unit
        self.condition = None
        self.setattr('out')

    def set_children(self):
        self.children['input'] = linear_bias(self.unit_dim)
        self.children['gate'] = linear_bias(self.unit_dim * 2)

    def init_params(self):
        self.u = self.allocate(orthogonal, 'U', weight, self.unit_dim, self.unit_dim)
        self.ug = self.allocate(orthogonal, 'Ug', weight, self.unit_dim, self.unit_dim * 2)

    def prepare(self, X):
        self.state_before = [self.children['input'].feedforward(X),
                             self.children['gate'].feedforward(X)]
        if self.mask:
            self.state_before.append(self.x_mask)
        self.initiate_state = [T.zeros([self.n_samples, self.unit_dim], theano.config.floatX)]
        self.context = []
        self.n_steps = self.input.shape[0]

    def step(self, *args):
        x = args[0]
        xg = args[1]
        h_ = args[2]
        gate = xg + T.dot(h_, self.ug)
        gate = T.nnet.sigmoid(gate)
        z_gate = self.slice(gate, 0, self.unit_dim)
        z_gate = self.addops('z_gate', z_gate, dropout, False)
        r_gate = self.slice(gate, 1, self.unit_dim)
        r_gate = self.addops('r_gate', r_gate, dropout, False)
        h_c = T.tanh(x + r_gate * T.dot(h_, self.u))
        h = (1 - z_gate) * h_ + z_gate * h_c
        h = self.addops('hidden_unit', h, dropout)
        h = self.add_mask(h, args)
        return h


class lstm(sequential):
    def __init__(self, unit, mask=True, **kwargs):
        sequential.__init__(self, mask=mask, **kwargs)
        self.mask = mask
        self.out = 'all'
        self.unit_dim = unit
        self.condition = None
        self.full = False
        self.setattr('out')

    def set_children(self):
        self.children['input'] = linear_bias(self.unit_dim * 4)

    def init_params(self):
        self.u = self.allocate(orthogonal, 'U', weight, self.unit_dim, self.unit_dim * 4)

    def prepare(self, X):
        self.state_before = [self.children['input'].feedforward(X)]
        if self.mask:
            self.state_before.append(self.x_mask)
        self.initiate_state = [T.zeros([self.n_samples, self.unit_dim], theano.config.floatX),
                               T.zeros([self.n_samples, self.unit_dim], theano.config.floatX)]
        self.context = []
        self.n_steps = self.input.shape[0]

    def step(self, *args):
        x = args[0]
        h_ = args[1]
        c_ = args[2]
        gate = x + T.dot(h_, self.u)
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
        h = self.add_mask(h, args)
        return h, c


class encoder(sequential):
    def __init__(self, unit, core=gru, structure='bi', **kwargs):
        sequential.__init__(self, **kwargs)
        self.mask = False
        self.unit_dim = unit
        if structure == 'bi':
            self.unit_dim = unit * 2
        self.core = core
        self.structure = structure

    def set_children(self):
        if self.structure == 'single':
            self.children['forward'] = self.core(self.unit_dim, self.mask)
        elif self.structure == 'bi':
            self.children['forward'] = self.core(self.unit_dim / 2, self.mask)
            self.children['backward'] = self.core(self.unit_dim / 2, self.mask, go_backwards=True)

    def get_output(self, X):
        self.output = self.apply(X)
        self.get_context()

    def apply(self, X):
        if self.structure == 'single':
            return self.children['forward'].feedscan(X)
        elif self.structure == 'bi':
            fwd = self.children['forward'].feedscan(X)
            bwd = self.children['backward'].feedscan(X)
            return concatenate([fwd, bwd[::-1, :, :]], fwd.ndim - 1)


class attention(layer):
    def __init__(self, unit, s_dim, gated=False, **kwargs):
        layer.__init__(self, unit, **kwargs)
        self.s_dim = s_dim
        self.gated = gated

    def set_children(self):
        if self.gated:
            self.children['dec_s'] = linear(self.unit_dim * 2, in_dim=self.s_dim)
        else:
            self.children['dec_s'] = linear(self.unit_dim, in_dim=self.s_dim)
        self.children['combine'] = linear_bias(1, in_dim=self.unit_dim)

    def apply(self, X):
        s_ = X[0]
        pctx = X[1]
        mask = X[2]
        if not self.gated:
            att_layer_1 = T.tanh(pctx + self.children['dec_s'].feedforward(s_))
            att_layer_1 = self.addops('att_in', att_layer_1, dropout)
            att_layer_2 = self.children['combine'].feedforward(att_layer_1)
            shp = []
            for i in range(att_layer_2.ndim - 1):
                shp.append(att_layer_2.shape[i])
            att = att_layer_2.reshape(shp)

            eij = T.exp(att - att.max(0, keepdims=True))
            # mask.shape = maxlen * batchsize
            if eij.ndim == 3:
                # eij.shape = maxlen * batchsize * beamsize
                eij = eij * mask[:, :, None]
            elif eij.ndim == 2:
                # eij.shape = maxlen * batchsize
                eij = eij * mask
            else:
                print eij.ndim
                assert False
            aij = eij / eij.sum(0, keepdims=True)
            return aij
        else:
            pctx1 = self.slice(pctx, 0, self.unit_dim)
            pctx2 = self.slice(pctx, 1, self.unit_dim)
            patt = self.children['dec_s'].feedforward(s_)
            att1 = self.slice(patt, 0, self.unit_dim)
            att2 = self.slice(patt, 1, self.unit_dim)
            att_layer_1 = (pctx1 + att1) * T.nnet.sigmoid(pctx2 + att2)
            att_layer_2 = self.children['combine'].feedforward(att_layer_1)
            att = att_layer_2
            eij = T.exp(att - att.max(0, keepdims=True))
            if att.ndim == 4:
                eij = eij * mask[:, :, None, None]
            elif att.ndim == 3:
                eij = eij * mask[:, :, None]
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

    def apply(self, X):
        s = X[0]
        y_ = X[1]
        ctx = X[2]
        prob = self.children['recurent'].feedforward(s) + self.children['peek'].feedforward(y_) + self.children[
            'glimpse'].feedforward(ctx)
        if self.maxout:
            prob = self.children['maxout'].feedforward(prob)
        else:
            prob = T.tanh(prob)
        prob = self.addops('emit_gate', prob, dropout)
        prob = self.children['predict'].feedforward(prob)
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
                 state_initiate='mean', in_channel=2, is_maxout=True, is_gated_attention=False, **kwargs):
        sequential.__init__(self, **kwargs)
        self.unit_dim = unit
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.core = core
        self.in_channel = in_channel
        self.state_initiate = state_initiate
        self.attention_unit = unit
        self.maxout = is_maxout
        self.is_gated_attention = is_gated_attention
        self.random_sample = True
        self.greedy = False
        self.y = None
        self.setattr('beam_size')
        self.setattr('attention_unit')

    def set_children(self):
        self.children['state_dec'] = self.core(self.unit_dim, mask=False, in_dim=self.emb_dim)
        self.children['glimpse_dec'] = self.core(self.unit_dim, mask=False, in_dim=self.in_dim)
        self.children['emitter'] = emitter(self.vocab_size, self.in_dim, self.unit_dim, self.emb_dim,
                                           in_dim=self.unit_dim, is_maxout=self.maxout)
        if not self.is_gated_attention:
            self.children['context'] = linear_bias(self.attention_unit, in_dim=self.in_dim)
            self.children['attention'] = attention(self.attention_unit, self.unit_dim)
        else:
            self.children['context'] = linear_bias(self.attention_unit * 2, in_dim=self.in_dim)
            self.children['attention'] = attention(self.attention_unit, self.unit_dim, self.in_dim)
        self.children['peek'] = lookuptable(self.emb_dim, in_dim=self.vocab_size)

    def init_params(self):
        self.wt_iniate_s = self.allocate(uniform, 'Wt_iniate_s', weight, self.in_dim // self.in_channel, self.unit_dim)

    def prepare(self, X):
        emb_y = self.children['peek'].feedforward(self.y)
        emb_shifted = T.zeros_like(emb_y)
        emb_shifted = T.set_subtensor(emb_shifted[1:], emb_y[:-1])
        self.y_ = emb_shifted
        state_before_y = self.children['state_dec'].children['input'].feedforward(emb_shifted)
        self.state_before = [state_before_y]
        if self.core == gru:
            state_before_yg_ = self.children['state_dec'].children['gate'].feedforward(emb_shifted)
            self.state_before = [state_before_y, state_before_yg_]

        mean_ctx = (self.input[:, :, -(self.in_dim // self.in_channel):] * self.x_mask[:, :, None]).sum(
            0) / self.x_mask.sum(0)[:, None]
        s_0 = mean_ctx
        if self.state_initiate == 'final': s_0 = self.input[0, :, -self.in_dim // self.in_channel:]
        self.s_0 = T.tanh(T.dot(s_0, self.wt_iniate_s))
        self.initiate_state = [self.s_0]
        if self.core == lstm:
            self.initiate_state.append(T.zeros([self.n_samples, self.unit_dim], theano.config.floatX))
        self.initiate_state.append(None)

        self.context = [self.children['context'].feedforward(self.input), self.input, self.x_mask]

        self.n_steps = self.y.shape[0]

    def get_output(self, X):
        self.get_n_samples()
        self.prepare(X)
        if self.core == lstm:
            [self.s, self.c, self.ci], updates = self.scan()
        else:
            [self.s, self.ci], updates = self.scan()
        self.pred = self.children['emitter'].feedforward([self.s, self.y_, self.ci])
        self.output = self.pred.reshape([self.y.shape[0], self.n_samples, self.vocab_size])
        self.updates.update(updates)

    def step_lstm(self, y_, s_, c_, pctx, ctx, x_m):
        s1, c1 = self.children['state_dec'].step(y_, s_, c_)
        aij = self.children['attention'].feedforward([s1, pctx, x_m])

        if aij.ndim == 2:
            ci = (ctx * aij[:, :, None]).sum(0)
        else:
            assert aij.ndim == 3
            ci = (ctx[:, :, None, :] * aij[:, :, :, None]).sum(0)

        condition = self.children['glimpse_dec'].children['input'].feedforward(ci)
        s2, c2 = self.children['glimpse_dec'].step(condition, s1, c1)
        s = s2
        c = c2
        return s, c, ci

    def step_gru(self, y_, yg_, s_, pctx, ctx, x_m):
        s1 = self.children['state_dec'].step(y_, yg_, s_)
        aij = self.children['attention'].feedforward([s1, pctx, x_m])

        if aij.ndim == 2:
            ci = (ctx * aij[:, :, None]).sum(0)
        else:
            assert aij.ndim == 3
            ci = (ctx * aij[:, :, :, None]).sum(0)

        condition = self.children['glimpse_dec'].children['input'].feedforward(ci)
        conditiong = self.children['glimpse_dec'].children['gate'].feedforward(ci)
        s2 = self.children['glimpse_dec'].step(condition, conditiong, s1)
        s = s2
        return s, ci

    def step(self, *args):
        if self.core == lstm:
            return self.step_lstm(*args)
        elif self.core == gru:
            return self.step_gru(*args)

    def get_predict(self):
        self.y_0 = T.zeros([self.n_samples, self.emb_dim])
        y_mm_0 = T.ones([self.n_samples], 'int64')
        initiate_state = [self.y_0, self.s_0, y_mm_0]
        if self.core == lstm:
            initiate_state.append(T.zeros([self.n_samples, self.unit_dim], theano.config.floatX))
        initiate_state.append(None)

        context = self.context

        def step_lstm(y_emb_, s_, c_, y_mm_, pctx, ctx, x_m):
            yi = self.children['state_dec'].children['input'].feedforward(y_emb_)
            s, c, ci = self.step(yi, s_, c_, pctx, ctx, x_m)

            prob = self.children['emitter'].feedforward([s, y_emb_, ci])
            prob = self.trng.multinomial(pvals=prob)
            pred = prob.argmax(-1)
            y_emb = self.children['peek'].feedforward(pred).reshape([self.n_samples, self.emb_dim])
            if not self.greedy:
                return [y_emb, s, T.ones([self.n_samples], 'int64'), pred]
            else:
                y_mm = y_mm_ * T.switch(T.eq(pred, 0), 0, 1)
                return [y_emb, s, y_mm, pred], theano.scan_module.until(T.eq(y_mm.sum(), 0))

        def step_gru(y_emb_, s_, y_mm_, pctx, ctx, x_m):
            yi = self.children['state_dec'].children['input'].feedforward(y_emb_)
            ygi = self.children['state_dec'].children['gate'].feedforward(y_emb_)
            s, ci = self.step(yi, ygi, s_, pctx, ctx, x_m)

            prob = self.children['emitter'].feedforward([s, y_emb_, ci])
            if self.random_sample:
                prob = self.trng.multinomial(pvals=prob)
            pred = prob.argmax(-1)
            y_emb = self.children['peek'].feedforward(pred).reshape([self.n_samples, self.emb_dim])
            if not self.greedy:
                return [y_emb, s, T.ones([self.n_samples], 'int64'), pred]
            else:
                y_mm = y_mm_ * T.switch(T.eq(pred, 0), 0, 1)
                return [y_emb, s, y_mm, pred], theano.scan_module.until(T.eq(y_mm.sum(), 0))

        if self.core == lstm:
            step = step_lstm
        else:
            step = step_gru

        if not self.greedy:

            result, updates = theano.scan(step, sequences=[], outputs_info=initiate_state,
                                          non_sequences=context, n_steps=self.y.shape[0])

            self.raw_updates = updates

            if self.core == lstm:
                y_emb, s, c, pred = result
            else:
                y_emb, s, y_mm, pred = result

            self.predict = pred
        else:
            result, updates = theano.scan(step, sequences=[], outputs_info=initiate_state,
                                          non_sequences=context, n_steps=self.y.shape[0])

            self.sample_updates = updates

            if self.core == lstm:
                y_emb, s, c, pred = result
            else:
                y_emb, s, y_mm, pred = result

            self.sample = T.transpose(pred)

    def gen_sample(self, beam_size=12):
        self.beam_size = beam_size
        if beam_size == 1:
            rs=self.random_sample
            self.random_sample=False
            self.greedy = True
            self.get_predict()
            self.greedy = False
            self.random_sample=rs
            return

        y_0 = T.zeros([self.n_samples, self.beam_size, self.emb_dim])
        y_mm_0 = T.ones([self.n_samples, self.beam_size], 'int8')
        s_0 = self.s_0[:, None, :]
        prob_sum_0 = T.zeros([self.n_samples, self.beam_size])
        initiate_state = [y_0, s_0]
        if self.core == lstm:
            c_0 = T.zeros([self.n_samples, self.beam_size, self.unit_dim], theano.config.floatX)
            initiate_state.append(c_0)
        initiate_state.extend([prob_sum_0, y_mm_0, T.constant(0), None, None, None, None, None])

        inp=self.input[:, :, None, :]

        context = [self.children['context'].feedforward(inp),inp,
                   self.x_mask]

        def initstate():
            y0 = self.children['state_dec'].children['input'].feedforward(y_0)
            if self.core == gru:
                yg0 = self.children['state_dec'].children['gate'].feedforward(y_0)
                s, ci = self.step(y0, yg0, s_0, context[0], context[1], self.x_mask)
                return s
            elif self.core == lstm:
                s, c, ci = self.step(y0, s_0, c_0, context[0], context[1], self.x_mask)
                return s, c, ci

        def lstmonestep(y_, s_):
            yi = self.children['state_dec'].children['input'].feedforward(y_)
            ygi = self.children['state_dec'].children['gate'].feedforward(y_)
            s, ci = self.step(yi, ygi, s_, context[0], context[1], self.x_mask)
            prob_bk_v = self.children['emitter'].feedforward([s, y_, ci])
            return s, ci, prob_bk_v

        def gruonestep(y_, s_, c_):
            yi = self.children['state_dec'].children['input'].feedforward(y_)
            s, c, ci = self.step(yi, s_, c_, context[0], context[1], self.x_mask)
            prob_bk_v = self.children['emitter'].feedforward([s, y_, ci])
            return s, c, ci, prob_bk_v

        from theano.ifelse import ifelse

        self.initstate = initstate
        self.gruonestep = gruonestep
        self.lstmonestep = gruonestep

        def step_lstm(y_emb_, s_, c_, prob_sum_b_k_, y_mm_b_k_, idx_1_, pctx, ctx, x_m):
            '''
            b:batch_size
            k:beam_size
            v:vocabe_size
            m:context_length(max_length)
            e:embedding_size
            u:n_hidden_units
            :return:
            :param y_emb_: (b,k,e)
            :param s_: (b,k,u)
            :param y_mm_b_k_: (b,k)
            :param idx_1_: (1)
            :param prob_sum_b_k_: (b,k)
            :return:
            '''
            yi = self.children['state_dec'].children['input'].feedforward(y_emb_)
            s,c, ci = self.step(yi, s_, c_, pctx, ctx, x_m)
            sss = s
            prob_bk_v = self.children['emitter'].feedforward([s, y_emb_, ci])

            prob_bk_v = -T.log(prob_bk_v * y_mm_b_k_.flatten()[:, None]) + prob_sum_b_k_.flatten()[:, None]
            prob_b_kv = ifelse(T.eq(idx_1_, 0), prob_bk_v[::self.beam_size, :],
                               prob_bk_v.reshape([self.n_samples, self.beam_size * self.vocab_size]))
            idx_1 = idx_1_ + 1
            y_b_k = T.argsort(prob_b_kv)[:, :self.beam_size]

            y_mod = y_b_k % self.vocab_size
            y_trans = (y_b_k // self.vocab_size + (T.arange(self.n_samples) * self.beam_size)[:, None]).flatten()
            s = T.reshape(s.reshape([self.n_samples * self.beam_size, self.unit_dim])[T.cast(y_trans, 'int64')],
                          [self.n_samples, self.beam_size, self.unit_dim])
            y_emb = T.reshape(self.children['peek'].feedforward(T.cast(y_mod.flatten(), 'int64')),
                              [self.n_samples, self.beam_size, self.emb_dim])

            y_mm_c_b_k = T.switch(T.eq(y_mod, 0), 0, 1)
            n_one = T.cast(y_mm_b_k_.sum(-1), 'int32')
            y_mm_tmp = T.arange(self.beam_size)[None, :]
            y_mm_shifted_b_k = T.switch(y_mm_tmp < n_one[:, None], 1, 0)
            prob_b_k = prob_b_kv[
                T.repeat(T.arange(self.n_samples), self.beam_size), y_b_k.flatten()].reshape(
                [self.n_samples, self.beam_size])
            y_mm = y_mm_c_b_k * y_mm_shifted_b_k
            return [y_emb, s, c, prob_b_k, y_mm, idx_1, y_mm_shifted_b_k, y_b_k, y_mod], theano.scan_module.until(
                T.eq(y_mm.sum(), 0))

        def step_gru(y_emb_, s_, prob_sum_b_k_, y_mm_b_k_, idx_1_, pctx, ctx, x_m):
            '''
            b:batch_size
            k:beam_size
            v:vocabe_size
            m:context_length(max_length)
            e:embedding_size
            u:n_hidden_units
            :return:
            :param y_emb_: (b,k,e)
            :param s_: (b,k,u)
            :param y_mm_b_k_: (b,k)
            :param idx_1_: (1)
            :param prob_sum_b_k_: (b,k)
            :return:
            '''
            yi = self.children['state_dec'].children['input'].feedforward(y_emb_)
            ygi = self.children['state_dec'].children['gate'].feedforward(y_emb_)
            s, ci = self.step(yi, ygi, s_, pctx, ctx, x_m)
            sss = s
            prob_bk_v = self.children['emitter'].feedforward([s, y_emb_, ci])

            prob_bk_v = -T.log(prob_bk_v * y_mm_b_k_.flatten()[:, None]) + prob_sum_b_k_.flatten()[:, None]
            prob_b_kv = ifelse(T.eq(idx_1_, 0), prob_bk_v[::self.beam_size, :],
                               prob_bk_v.reshape([self.n_samples, self.beam_size * self.vocab_size]))
            idx_1 = idx_1_ + 1
            y_b_k = T.argsort(prob_b_kv)[:, :self.beam_size]

            y_mod = y_b_k % self.vocab_size
            y_trans = (y_b_k // self.vocab_size + (T.arange(self.n_samples) * self.beam_size)[:, None]).flatten()
            s = T.reshape(s.reshape([self.n_samples * self.beam_size, self.unit_dim])[T.cast(y_trans, 'int64')],
                          [self.n_samples, self.beam_size, self.unit_dim])
            y_emb = T.reshape(self.children['peek'].feedforward(T.cast(y_mod.flatten(), 'int64')),
                              [self.n_samples, self.beam_size, self.emb_dim])

            y_mm_c_b_k = T.switch(T.eq(y_mod, 0), 0, 1)
            n_one = T.cast(y_mm_b_k_.sum(-1), 'int32')
            y_mm_tmp = T.arange(self.beam_size)[None, :]
            y_mm_shifted_b_k = T.switch(y_mm_tmp < n_one[:, None], 1, 0)
            prob_b_k = prob_b_kv[
                T.repeat(T.arange(self.n_samples), self.beam_size), y_b_k.flatten()].reshape(
                [self.n_samples, self.beam_size])
            y_mm = y_mm_c_b_k * y_mm_shifted_b_k
            return [y_emb, s, prob_b_k, y_mm, idx_1, y_mm_shifted_b_k, y_trans, y_mod], theano.scan_module.until(
                T.eq(y_mm.sum(), 0))

        if self.core == lstm:
            step = step_lstm
        else:
            step = step_gru

        result, updates = theano.scan(step, sequences=[], outputs_info=initiate_state,
                                      non_sequences=context, n_steps=50)

        self.result = result

        self.gen_updates = updates

        if self.core == lstm:
            self.y_emb, self.s, self.c, self.prob, self.y_mm, self.idx, self.y_mm_shifted, self.y_trans, self.y_mod = result
        else:
            self.y_emb, self.s, self.prob, self.y_mm, self.idx, self.y_mm_shifted, self.y_trans, self.y_mod = result

        def dec(mm, ms, t_trans, y_mod, y_prob, y_idx_):
            def d(mm, ms, y_idx_):
                msm = T.xor(mm, ms)
                lc = mm.sum()
                ls = msm.sum()
                eid = T.argsort(msm)[::-1]
                y_idx = T.set_subtensor(y_idx_[lc:lc + ls], eid[:ls])
                score_mark = T.set_subtensor(T.zeros([beam_size])[lc:lc + ls], 1)
                return y_idx, score_mark

            [y_idx_c, score_mark], u = theano.scan(d, [mm, ms, y_idx_])
            ax0 = T.repeat(T.arange(self.n_samples), self.beam_size)
            y_prob_shift = y_prob[ax0, y_idx_c.flatten()].reshape([self.n_samples, self.beam_size])
            score = T.switch(T.eq(score_mark, 1), y_prob_shift, T.zeros([beam_size]))
            t_trans = t_trans[ax0, y_idx_c.flatten()].reshape([self.n_samples, self.beam_size])
            t_mod = y_mod[ax0, y_idx_c.flatten()].reshape([self.n_samples, self.beam_size])
            y_out = T.switch(T.eq(ms, 1), t_mod, 0)
            y_idx = T.switch(T.eq(ms, 1), t_trans, 0)

            return y_idx, y_out, score, y_idx_c

        idx_list_0 = T.zeros([self.n_samples, self.beam_size], 'int64')
        if self.beam_size == 1:
            idx_list_0 = T.unbroadcast(idx_list_0, 1)
        [self.y_idx, self.y_out, self.score, self.y_idx_c], decupdates = theano.scan(dec, sequences=[self.y_mm,
                                                                                                  self.y_mm_shifted,
                                                                                                  self.y_trans,
                                                                                                  self.y_mod,
                                                                                                  self.prob],
                                                                                  outputs_info=[idx_list_0, None, None,
                                                                                                None],
                                                                                  go_backwards=True)
        self.sample_updates=self.gen_updates
        self.sample_updates.update(decupdates)
        self.score_sum = (self.score / T.arange(self.score.shape[0], 0, -1, theano.config.floatX)[:, None, None]).sum(0)
        self.choice = self.score_sum.argmin(-1)
        self.samples = self.y_out.dimshuffle(1, 2, 0)[:, :, ::-1]
        self.sample = self.samples[T.arange(self.n_samples), self.choice]

    def gen_sample_debug(self, beam_size=12):
        self.beam_size = beam_size
        if beam_size == 1:
            rs=self.random_sample
            self.random_sample=False
            self.greedy = True
            self.get_predict()
            self.greedy = False
            self.random_sample=rs
            return

        y_0 = T.zeros([self.n_samples, self.beam_size, self.emb_dim])
        y_mm_0 = T.ones([self.n_samples, self.beam_size], 'int8')
        s_0 = T.tile(self.s_0[:, None, :], [1, self.beam_size, 1])
        prob_sum_0 = T.zeros([self.n_samples, self.beam_size])
        initiate_state = [y_0, s_0]
        if self.core == lstm:
            c_0 = T.zeros([self.n_samples, self.beam_size, self.unit_dim], theano.config.floatX)
            initiate_state.append(c_0)
        initiate_state.extend([prob_sum_0, y_mm_0, T.constant(0), None, None, None, None, None])

        inp=T.tile(self.input[:, :, None, :], [1, 1, self.beam_size, 1])

        context = [self.children['context'].feedforward(inp),inp,
                   self.x_mask]

        def initstate():
            y0 = self.children['state_dec'].children['input'].feedforward(y_0)
            if self.core == gru:
                yg0 = self.children['state_dec'].children['gate'].feedforward(y_0)
                s, ci = self.step(y0, yg0, s_0, context[0], context[1], self.x_mask)
                return s
            elif self.core == lstm:
                s, c, ci = self.step(y0, s_0, c_0, context[0], context[1], self.x_mask)
                return s, c, ci

        def lstmonestep(y_, s_):
            yi = self.children['state_dec'].children['input'].feedforward(y_)
            ygi = self.children['state_dec'].children['gate'].feedforward(y_)
            s, ci = self.step(yi, ygi, s_, context[0], context[1], self.x_mask)
            prob_bk_v = self.children['emitter'].feedforward([s, y_, ci])
            return s, ci, prob_bk_v

        def gruonestep(y_, s_, c_):
            yi = self.children['state_dec'].children['input'].feedforward(y_)
            s, c, ci = self.step(yi, s_, c_, context[0], context[1], self.x_mask)
            prob_bk_v = self.children['emitter'].feedforward([s, y_, ci])
            return s, c, ci, prob_bk_v

        from theano.ifelse import ifelse

        self.initstate = initstate
        self.gruonestep = gruonestep
        self.lstmonestep = gruonestep

        def step_lstm(y_emb_, s_, c_, prob_sum_b_k_, y_mm_b_k_, idx_1_, pctx, ctx, x_m):
            '''
            b:batch_size
            k:beam_size
            v:vocabe_size
            m:context_length(max_length)
            e:embedding_size
            u:n_hidden_units
            :return:
            :param y_emb_: (b,k,e)
            :param s_: (b,k,u)
            :param y_mm_b_k_: (b,k)
            :param idx_1_: (1)
            :param prob_sum_b_k_: (b,k)
            :return:
            '''
            yi = self.children['state_dec'].children['input'].feedforward(y_emb_)
            s, ci = self.step(yi, s_, c_, pctx, ctx, x_m)
            sss = s
            prob_bk_v = self.children['emitter'].feedforward([s, y_emb_, ci])

            prob_bk_v = -T.log(prob_bk_v * y_mm_b_k_.flatten()[:, None]) + prob_sum_b_k_.flatten()[:, None]
            prob_b_kv = ifelse(T.eq(idx_1_, 0), prob_bk_v[::self.beam_size, :],
                               prob_bk_v.reshape([self.n_samples, self.beam_size * self.vocab_size]))
            idx_1 = idx_1_ + 1
            y_b_k = T.argsort(prob_b_kv)[:, :self.beam_size]

            y_mod = y_b_k % self.vocab_size
            y_trans = (y_b_k // self.vocab_size + (T.arange(self.n_samples) * self.beam_size)[:, None]).flatten()
            s = T.reshape(s.reshape([self.n_samples * self.beam_size, self.unit_dim])[T.cast(y_trans, 'int64')],
                          [self.n_samples, self.beam_size, self.unit_dim])
            y_emb = T.reshape(self.children['peek'].feedforward(T.cast(y_mod.flatten(), 'int64')),
                              [self.n_samples, self.beam_size, self.emb_dim])

            y_mm_c_b_k = T.switch(T.eq(y_mod, 0), 0, 1)
            n_one = T.cast(y_mm_b_k_.sum(-1), 'int32')
            y_mm_tmp = T.arange(self.beam_size)[None, :]
            y_mm_shifted_b_k = T.switch(y_mm_tmp < n_one[:, None], 1, 0)
            prob_b_k = prob_b_kv[
                T.repeat(T.arange(self.n_samples), self.beam_size), y_b_k.flatten()].reshape(
                [self.n_samples, self.beam_size])
            y_mm = y_mm_c_b_k * y_mm_shifted_b_k
            return [y_emb, s, c, prob_b_k, y_mm, idx_1, y_mm_shifted_b_k, y_b_k, y_mod, y_trans,
                    sss], theano.scan_module.until(
                T.eq(y_mm.sum(), 0))

        def step_gru(y_emb_, s_, prob_sum_b_k_, y_mm_b_k_, idx_1_, pctx, ctx, x_m):
            '''
            b:batch_size
            k:beam_size
            v:vocabe_size
            m:context_length(max_length)
            e:embedding_size
            u:n_hidden_units
            :return: 
            :param y_emb_: (b,k,e)
            :param s_: (b,k,u)
            :param y_mm_b_k_: (b,k)
            :param idx_1_: (1)
            :param prob_sum_b_k_: (b,k)
            :return: 
            '''
            yi = self.children['state_dec'].children['input'].feedforward(y_emb_)
            ygi = self.children['state_dec'].children['gate'].feedforward(y_emb_)
            s, ci = self.step(yi, ygi, s_, pctx, ctx, x_m)
            sss = s
            prob_bk_v = self.children['emitter'].feedforward([s, y_emb_, ci])

            prob_bk_v = -T.log(prob_bk_v * y_mm_b_k_.flatten()[:, None]) + prob_sum_b_k_.flatten()[:, None]
            prob_b_kv = ifelse(T.eq(idx_1_, 0), prob_bk_v[::self.beam_size, :],
                               prob_bk_v.reshape([self.n_samples, self.beam_size * self.vocab_size]))
            idx_1 = idx_1_ + 1
            y_b_k = T.argsort(prob_b_kv)[:, :self.beam_size]

            y_mod = y_b_k % self.vocab_size
            y_trans = (y_b_k // self.vocab_size + (T.arange(self.n_samples) * self.beam_size)[:, None]).flatten()
            s = T.reshape(s.reshape([self.n_samples * self.beam_size, self.unit_dim])[T.cast(y_trans, 'int64')],
                          [self.n_samples, self.beam_size, self.unit_dim])
            y_emb = T.reshape(self.children['peek'].feedforward(T.cast(y_mod.flatten(), 'int64')),
                              [self.n_samples, self.beam_size, self.emb_dim])

            y_mm_c_b_k = T.switch(T.eq(y_mod, 0), 0, 1)
            n_one = T.cast(y_mm_b_k_.sum(-1), 'int32')
            y_mm_tmp = T.arange(self.beam_size)[None, :]
            y_mm_shifted_b_k = T.switch(y_mm_tmp < n_one[:, None], 1, 0)
            prob_b_k = prob_b_kv[
                T.repeat(T.arange(self.n_samples), self.beam_size), y_b_k.flatten()].reshape(
                [self.n_samples, self.beam_size])
            y_mm = y_mm_c_b_k * y_mm_shifted_b_k
            return [y_emb, s, prob_b_k, y_mm, idx_1, y_mm_shifted_b_k, y_b_k, y_mod, y_trans,
                    sss], theano.scan_module.until(
                T.eq(y_mm.sum(), 0))

        if self.core == lstm:
            step = step_lstm
        else:
            step = step_gru

        result, updates = theano.scan(step, sequences=[], outputs_info=initiate_state,
                                      non_sequences=context, n_steps=50)

        self.result = result

        self.gen_updates = updates

        if self.core == lstm:
            self.y_emb, self.s, self.c, self.prob, self.y_mm, self.idx, self.y_mm_shifted, self.y_pred, self.y_mod, self.s_c, self.sss = result
        else:
            self.y_emb, self.s, self.prob, self.y_mm, self.idx, self.y_mm_shifted, self.y_pred, self.y_mod, self.s_c, self.sss = result

        def dec(mm, ms, y_pred, y_mod, y_prob, y_idx_):
            def d(mm, ms, y_idx_):
                msm = T.xor(mm, ms)
                lc = mm.sum()
                ls = msm.sum()
                eid = T.argsort(msm)[::-1]
                y_idx = T.set_subtensor(y_idx_[lc:lc + ls], eid[:ls])
                score_mark = T.set_subtensor(T.zeros([beam_size])[lc:lc + ls], 1)
                return y_idx, score_mark

            [y_idx_c, score_mark], u = theano.scan(d, [mm, ms, y_idx_])
            ax0 = T.repeat(T.arange(self.n_samples), self.beam_size)
            y_prob_shift = y_prob[ax0, y_idx_c.flatten()].reshape([self.n_samples, self.beam_size])
            score = T.switch(T.eq(score_mark, 1), y_prob_shift, T.zeros([beam_size]))
            y_pred = y_pred[ax0, y_idx_c.flatten()].reshape([self.n_samples, self.beam_size])
            y_mod = y_mod[ax0, y_idx_c.flatten()].reshape([self.n_samples, self.beam_size])
            y_before = y_pred // self.vocab_size
            y_out = T.switch(T.eq(ms, 1), y_mod, 0)
            y_idx = T.switch(T.eq(ms, 1), y_before, 0)

            return y_idx, y_out, score, y_idx_c

        idx_list_0 = T.zeros([self.n_samples, self.beam_size], 'int64')
        if self.beam_size == 1:
            idx_list_0 = T.unbroadcast(idx_list_0, 1)
        [self.y_idx, self.y_out, self.score, self.y_idx_c], updates = theano.scan(dec, sequences=[self.y_mm,
                                                                                                  self.y_mm_shifted,
                                                                                                  self.y_pred,
                                                                                                  self.y_mod,
                                                                                                  self.prob],
                                                                                  outputs_info=[idx_list_0, None, None,
                                                                                                None],
                                                                                  go_backwards=True)
        self.sample_updates = OrderedDict()
        self.sample_updates.update(self.gen_updates)
        self.sample_updates.update(self.updates)
        self.score_sum = (self.score / T.arange(self.score.shape[0], 0, -1, theano.config.floatX)[:, None, None]).sum(0)
        self.choice = self.score_sum.argmin(-1)
        self.samples = self.y_out.dimshuffle(1, 2, 0)[:, :, ::-1]
        self.sample = self.samples[T.arange(self.n_samples), self.choice]

    def get_cost(self, Y):
        cost = T.nnet.categorical_crossentropy(self.pred, Y.flatten())
        cost = cost.reshape([Y.shape[0], Y.shape[1]])
        cost = (cost * self.y_mask).sum(0)
        self.cost = T.mean(cost)

    def get_error(self, Y):
        self.error = T.sum(T.neq(Y, self.predict) * self.y_mask) / T.sum(self.y_mask)


class cnn_encoder(sequential):
    pass


class cnn_decoder(sequential):
    pass
