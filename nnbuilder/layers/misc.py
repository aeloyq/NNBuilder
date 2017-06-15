# -*- coding: utf-8 -*-
"""
Created on  四月 27 23:10 2017

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
from sequential import *
class rgru(sequential):
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
        self.children['gate'] = linear_bias(self.unit_dim * 3)

    def init_params(self):
        self.u = self.allocate(orthogonal, 'U', weight, self.unit_dim, self.unit_dim)
        self.ug = self.allocate(orthogonal, 'Ug', weight, self.unit_dim, self.unit_dim * 3)

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
            z_gate = T.nnet.relu(self.slice(gate, 0, self.unit_dim))+1e-8
            z_gate = self.addops('z_gate', z_gate, dropout, False)
            zp_gate = T.nnet.relu(self.slice(gate, 1, self.unit_dim))+1e-8
            zp_gate = self.addops('zp_gate', zp_gate, dropout, False)
            r_gate = T.nnet.sigmoid(self.slice(gate, 2, self.unit_dim))
            r_gate = self.addops('r_gate', r_gate, dropout, False)
            h_c = T.tanh(x + T.dot(r_gate * h_, u))
            z_gate_sum=z_gate+zp_gate
            h = (zp_gate/z_gate_sum) * h_ + (z_gate/z_gate_sum) * h_c
            h = self.addops('hidden_unit', h, dropout)
            return h

        def step_mask(x, xg, m, h_, u, ug):
            gate = xg + T.dot(h_, ug)
            gate = self.conditional(gate)
            z_gate = T.nnet.relu(self.slice(gate, 0, self.unit_dim))+1e-8
            z_gate = self.addops('z_gate', z_gate, dropout, False)
            zp_gate = T.nnet.relu(self.slice(gate, 1, self.unit_dim))+1e-8
            zp_gate = self.addops('zp_gate', zp_gate, dropout, False)
            r_gate = T.nnet.sigmoid(self.slice(gate, 2, self.unit_dim))
            r_gate = self.addops('r_gate', r_gate, dropout, False)
            h_c = T.tanh(x + T.dot(r_gate * h_, u))
            z_gate_sum=z_gate+zp_gate
            h = (zp_gate/z_gate_sum) * h_ + (z_gate/z_gate_sum) * h_c
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

class mencoder(sequential):
    def __init__(self, unit, core=gru, num=1, **kwargs):
        sequential.__init__(self, **kwargs)
        self.unit_dim = unit
        self.core = core
        self.num=num

    def set_children(self):
        self.children['forward'] = self.core(self.unit_dim, self.mask)
        self.children['backward'] = self.core(self.unit_dim, self.mask, go_backwards=True)

    def get_output(self, X, P):
        self.output = self.apply(X, P)

    def apply(self, X, P):
        fwd = self.children['forward'].feedscan(X, P)
        bwd = self.children['backward'].feedscan(X, P)
        return concatenate([fwd, bwd[::-1]], fwd.ndim - 1)

class mdecoder(sequential):
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
            self.children['attention'] = attention(self.attention_unit, self.unit_dim, self.in_dim * self.be)
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
        self.s_0 = s_0
        if self.core== lstm:
            self.initiate_state.append(T.zeros([self.n_samples, self.unit_dim], theano.config.floatX))
        self.initiate_state.append(None)

        self.context = [self.children['context'].feedforward(self.input), self.input]
        self.context.append(self.x_mask)
        plist = []
        if self.core == lstm:
            plist = ['U_state_dec', 'Wt_attention_dec_s', 'Wt_attention_combine', 'Bi_attention_combine',
                     'Wt_glimpse_dec_input', 'Bi_glimpse_dec_input', 'U_glimpse_dec']
        else:
            plist = ['U_state_dec', 'Ug_state_dec', 'Wt_attention_dec_s', 'Wt_attention_combine',
                     'Bi_attention_combine', 'Wt_glimpse_dec_input', 'Bi_glimpse_dec_input',
                     'Wt_glimpse_dec_gate', 'Bi_glimpse_dec_gate', 'U_glimpse_dec', 'Ug_glimpse_dec']
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
                if aij.ndim == 2:
                    ci = (ctx * aij[:, :, None]).sum(0)
                elif aij.ndim == 3:
                    ci = (ctx[:, :, None, :] * aij[:, :, :, None]).sum(0)
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
                if aij.ndim == 2:
                    ci = (ctx * aij[:, :, None]).sum(0)
                elif aij.ndim == 3:

                    ci = (ctx[:, :, None, :] * aij[:, :, :, None]).sum(0)
            else:
                ci = (ctx * aij).sum(0)

            condition = T.dot(ci, gwt) + gbi
            conditiong = T.dot(ci, gwtg) + gbig
            s2 = self.children['glimpse_dec'].feedstep()(condition, conditiong, s1, gu, gug)
            s = s2
            return s, ci

        if self.core == lstm:
            return step_lstm
        else:
            return step_gru

    def scan(self, step):
        return theano.scan(step, sequences=self.state_before, outputs_info=self.initiate_state,
                           non_sequences=self.context, n_steps=self.n_steps, strict=True)

    def get_predict(self):
        y_0 = T.zeros([self.n_samples, self.emb_dim])
        s_0 = self.s_0
        initiate_state = [y_0, s_0]
        if self.core==lstm:
            initiate_state.append(T.zeros([self.n_samples, self.unit_dim], theano.config.floatX))
        initiate_state.append(None)

        context = [self.children['context'].feedforward(self.input), self.input]
        context.append(self.x_mask)
        plist = []
        if self.core == lstm:
            plist = ['Wt_state_dec_input', 'Bi_state_dec_input', 'U_state_dec', 'Wt_attention_dec_s',
                     'Wt_attention_combine', 'Bi_attention_combine',
                     'Wt_glimpse_dec_input', 'Bi_glimpse_dec_input', 'U_glimpse_dec',
                     'Wt_emitter_recurent', 'Bi_emitter_recurent', 'Wt_emitter_peek', 'Wt_emitter_glimpse',
                     'Wt_emitter_predict', 'Wemb_peek']
        else:
            plist = ['Wt_state_dec_input', 'Bi_state_dec_input', 'Wt_state_dec_gate', 'Bi_state_dec_gate',
                     'U_state_dec', 'Ug_state_dec', 'Wt_attention_dec_s', 'Wt_attention_combine',
                     'Bi_attention_combine', 'Wt_glimpse_dec_input', 'Bi_glimpse_dec_input',
                     'Wt_glimpse_dec_gate', 'Bi_glimpse_dec_gate', 'U_glimpse_dec', 'Ug_glimpse_dec',
                     'Wt_emitter_recurent', 'Bi_emitter_recurent', 'Wt_emitter_peek', 'Wt_emitter_glimpse',
                     'Wt_emitter_predict', 'Wemb_peek']
        for i in plist:
            ii = i.split('_')
            iii = ii[0] + '_' + self.name
            for j in range(1, len(ii)):
                iii = iii + '_' + ii[j]
            context.append(self.params[iii])

        def step_lstm(y_emb_, s_, c_, pctx, ctx, x_m, swt, sbi, su, adwt, acwt, acbi, gwt, gbi, gu, erwt, erbi,
                      epwt, egwt, edwt, wemb):
            yi = T.dot(y_emb_, swt) + sbi
            s, c, ci = self.feedstep()(yi, s_, c_, pctx, ctx, x_m, su, adwt, acwt, acbi, gwt, gbi, gu)

            prob = self.children['emitter'].feedforward([s, y_emb_, ci], {
                'emitter': {'recurent': {'Wt': erwt, 'Bi': erbi}, 'peek': {'Wt': epwt}, 'glimpse': {'Wt': egwt},
                            'predict': {'Wt': edwt}}})
            prob = self.trng.multinomial(pvals=prob)
            pred = prob.argmax(-1)
            y_emb = T.reshape(wemb[pred], [self.n_samples, self.emb_dim])
            return [y_emb, s, c, pred]

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
            #prob = self.trng.multinomial(pvals=prob)
            pred = prob.argmax(-1)
            y_emb = T.reshape(wemb[pred], [self.n_samples, self.emb_dim])
            return [y_emb, s, pred]

        step = None
        if self.core == lstm:
            step = step_lstm
        else:
            step = step_gru

        result, updates = theano.scan(step, sequences=[], outputs_info=initiate_state,
                                      non_sequences=context, strict=True, n_steps=self.y.shape[0])
        updates.update(self.updates)

        self.raw_updates = updates

        pred = 0

        if self.core == lstm:
            y_emb, s, c, pred = result
        else:
            y_emb, s, pred = result

        self.predict = pred

    def gen_sample(self, beam_size=12):
        self.beam_size = beam_size
        y_0 = T.zeros([self.n_samples, self.beam_size, self.emb_dim])
        y_mm_0 = T.ones([self.n_samples, self.beam_size],'int8')
        if self.beam_size == 1:
            y_mm_0 = T.unbroadcast(y_mm_0, 1)
        s_0 = self.s_0
        s_0 = T.tile(s_0, self.beam_size).reshape([self.beam_size, self.n_samples, self.unit_dim])
        s_0 = s_0.dimshuffle(1, 0, 2)
        initiate_state = [y_0, s_0]
        if self.core==lstm:
            initiate_state.append(T.zeros([self.n_samples, self.beam_size, self.unit_dim], theano.config.floatX))
        i_y_prob = T.zeros([self.n_samples, self.beam_size])
        initiate_state.extend([y_mm_0, T.constant(0), i_y_prob, None, None, None, None])

        pctx = self.children['context'].feedforward(self.input)[:, :, None, :]
        ctx = self.input
        context = [pctx, ctx, self.x_mask]
        plist = []
        if self.core == lstm:
            plist = ['Wt_state_dec_input', 'Bi_state_dec_input', 'U_state_dec', 'Wt_attention_dec_s',
                     'Wt_attention_combine', 'Bi_attention_combine',
                     'Wt_glimpse_dec_input', 'Bi_glimpse_dec_input', 'U_glimpse_dec',
                     'Wt_emitter_recurent', 'Bi_emitter_recurent', 'Wt_emitter_peek', 'Wt_emitter_glimpse',
                     'Wt_emitter_predict', 'Wemb_peek']
        else:
            plist = ['Wt_state_dec_input', 'Bi_state_dec_input', 'Wt_state_dec_gate', 'Bi_state_dec_gate',
                     'U_state_dec', 'Ug_state_dec', 'Wt_attention_dec_s', 'Wt_attention_combine',
                     'Bi_attention_combine', 'Wt_glimpse_dec_input', 'Bi_glimpse_dec_input',
                     'Wt_glimpse_dec_gate', 'Bi_glimpse_dec_gate', 'U_glimpse_dec', 'Ug_glimpse_dec',
                     'Wt_emitter_recurent', 'Bi_emitter_recurent', 'Wt_emitter_peek', 'Wt_emitter_glimpse',
                     'Wt_emitter_predict', 'Wemb_peek']
        for i in plist:
            ii = i.split('_')
            iii = ii[0] + '_' + self.name
            for j in range(1, len(ii)):
                iii = iii + '_' + ii[j]
            context.append(self.params[iii])

        from theano.ifelse import ifelse

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

        def step_gru(y_emb_, s_, y_mm_b_k_, idx_1_, prob_sum_b_k_, pctx, ctx, x_m, swt, sbi, swtg, sbig, su, sug, adwt,
                     acwt, acbi, gwt,
                     gbi, gwtg,
                     gbig, gu, gug, erwt, erbi, epwt, egwt, edwt, wemb):
            '''
            b:batch_size
            k:beam_size
            v:vocabe_size
            m:context_length(max_length)
            e:embedding_size
            u:n_hidden_units
            :return: 
            :param y_emb_: (b,e)
            :param s_: (b,u)
            :param y_mm_b_k_: (b,k)
            :param idx_1_: (1)
            :param prob_sum_b_k_: (b,k)
            :return: 
            '''

            yi = T.dot(y_emb_, swt) + sbi
            ygi = T.dot(y_emb_, swtg) + sbig
            s, ci = self.feedstep()(yi, ygi, s_, pctx, ctx, x_m, su, sug, adwt, acwt, acbi, gwt, gbi, gwtg, gbig, gu,
                                    gug)
            prob = self.children['emitter'].feedforward([s, y_emb_, ci], {
                'emitter': {'recurent': {'Wt': erwt, 'Bi': erbi}, 'peek': {'Wt': epwt}, 'glimpse': {'Wt': egwt},
                            'predict': {'Wt': edwt}}})

            prob_bk_v = prob
            prob_b_kv = prob.reshape([self.n_samples, self.beam_size * self.vocab_size])

            prob_sum_bk_v = -T.log(prob_bk_v * y_mm_b_k_.flatten()[:, None]) + prob_sum_b_k_.flatten()[:, None]
            prob_sum_b_kv = ifelse(T.eq(idx_1_, 0), prob_sum_bk_v[::self.beam_size, :],
                                   prob_sum_bk_v.reshape([self.n_samples, self.beam_size * self.vocab_size]))
            idx_1 = idx_1_ + 1
            y_b_k = T.argsort(prob_sum_b_kv)[:, :self.beam_size]

            y_mod = y_b_k % self.vocab_size
            y_emb = T.reshape(wemb[T.cast(y_mod.flatten(), 'int64')], [self.n_samples, self.beam_size, self.emb_dim])

            y_mm_c_b_k = T.switch(T.eq(y_mod, 0), 0, 1)
            n_one = T.cast(y_mm_b_k_.sum(-1), 'int32')
            y_mm_tmp = T.arange(self.beam_size)[None, :]
            y_mm_shifted_b_k = T.switch(y_mm_tmp < n_one[:, None], 1, 0)
            prob_sum_b_k = prob_sum_b_kv[
                T.repeat(T.arange(self.n_samples), self.beam_size), y_b_k.flatten()].reshape(
                [self.n_samples, self.beam_size])
            y_mm = y_mm_c_b_k * y_mm_shifted_b_k
            prob_raw_b_k=prob_b_kv[
                T.repeat(T.arange(self.n_samples), self.beam_size), y_b_k.flatten()].reshape(
                [self.n_samples, self.beam_size])
            return [y_emb, s, y_mm, idx_1, prob_sum_b_k, y_mm_shifted_b_k, y_b_k, y_mod,
                    prob_raw_b_k], theano.scan_module.until(T.eq(y_mm.sum(), 0))

        step = None
        if self.core == lstm:
            step = step_lstm
        else:
            step = step_gru

        result, updates = theano.scan(step, sequences=[], outputs_info=initiate_state,
                                      non_sequences=context, strict=True, n_steps=50)

        self.result = result

        self.gen_updates = updates

        if self.core == lstm:
            y_emb, s, c, self.y_mm, is_first, self.y_mm_shifted, self.y_flat, self.y_mod, self.p_y_flat = result
        else:
            y_emb, s, self.y_mm, self.idx, self.prob_sum, self.y_mm_shifted, self.y_pred, self.y_mod, self.prob_y = result

        def dec(mm,ms, y_pred, y_mod,y_prob, y_idx_,n):
            def d(mm,ms,y_idx_,y_prob,n):
                msm=T.xor(mm,ms)
                lc = mm.sum()
                ls = msm.sum()
                eid=T.argsort(msm)[::-1]
                y_idx_ = T.set_subtensor(y_idx_[lc:lc + ls], eid[:ls])
                score=T.switch(T.eq(msm,1),y_prob/n,0)

                return y_idx_,score
            [y_idx_c,score],u=theano.scan(d,[mm,ms,y_idx_,y_prob],non_sequences=[n])

            ax0 = T.repeat(T.arange(self.n_samples), self.beam_size)
            y_pred = y_pred[ax0, y_idx_c.flatten()].reshape([self.n_samples, self.beam_size])
            y_mod = y_mod[ax0, y_idx_c.flatten()].reshape([self.n_samples, self.beam_size])

            y_before = y_pred // self.vocab_size
            y_out = T.switch(T.eq(ms, 1), y_mod, 0)
            y_idx = T.switch(T.eq(ms, 1), y_before,0)


            return y_idx,n-1, y_out,score,y_idx_c

        mm=self.y_mm
        idx_list_0=T.zeros([self.n_samples,self.beam_size],'int64')
        if self.beam_size==1:
            idx_list_0 =T.unbroadcast(idx_list_0,1)
        n=mm.shape[0]
        [self.y_idx, nn,self.y_out,self.score,self.y_idx_c], updates = theano.scan(dec, sequences=[self.y_mm,self.y_mm_shifted, self.y_pred, self.y_mod,self.prob_sum],
                                                    outputs_info=[idx_list_0,n, None,None,None], go_backwards=True)
        self.sample_updates = OrderedDict()
        self.sample_updates.update(self.gen_updates)
        self.sample_updates.update(self.updates)
        self.choice=self.score.sum(0).argmin(-1)
        self.samples=self.y_out.dimshuffle(1, 2, 0)[:, :, ::-1]
        self.sample = self.samples[T.arange(self.n_samples), self.choice]

    def get_cost(self, Y):
        cost = T.nnet.categorical_crossentropy(self.output_cost, Y.flatten())
        cost = cost.reshape([Y.shape[0], Y.shape[1]])
        cost = (cost * self.y_mask).sum(0)
        self.cost = T.mean(cost)

    def get_error(self, Y):
        self.error = T.sum(T.neq(Y, self.predict) * self.y_mask) / T.sum(self.y_mask)