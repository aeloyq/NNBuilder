# -*- coding: utf-8 -*-
"""
Created on  十一月 04 18:20 2017

@author: aeloyq
"""
# -*- coding: utf-8 -*-
"""
Created on  Feb 13 11:16 PM 2017

@author: aeloyq
"""
from nnbuilder.layers.basic import *
from nnbuilder.layers.simple import *
from nnbuilder.layers.recurrent import *
from nnbuilder.layers.misc import *


class encoder(rechidden):
    def __init__(self, unit, core=gru, structure='bi', **kwargs):
        rechidden.__init__(self, **kwargs)
        self.mask = False
        self.unit_dim = unit
        if structure == 'bi':
            self.unit_dim = unit * 2
        self.core = core
        self.structure = structure

    def set_sublayers(self):
        if self.structure == 'single':
            self.sublayers['forward'] = self.core(self.unit_dim, self.mask)
        elif self.structure == 'bi':
            self.sublayers['forward'] = self.core(self.unit_dim / 2, self.mask)
            self.sublayers['backward'] = self.core(self.unit_dim / 2, self.mask, go_backwards=True)

    def process_output(self, X):
        self.output = self.apply(X)
        self.extract_output()

    def apply(self, X):
        if self.structure == 'single':
            return self.sublayers['forward'].feedscan(X)
        elif self.structure == 'bi':
            fwd = self.sublayers['forward'].feedscan(X)
            bwd = self.sublayers['backward'].feedscan(X)
            return concatenate([fwd, bwd[::-1, :, :]], fwd.ndim - 1)


class decoder(rechidden):
    def __init__(self, unit, emb_dim, vocab_size, core=gru,
                 state_initiate='mean', in_channel=2, is_maxout=True, is_gated_attention=False, is_mutiple=False,
                 **kwargs):
        rechidden.__init__(self, **kwargs)
        self.unit_dim = unit
        self.emb_dim = emb_dim
        self.vocab_size = vocab_size
        self.core = core
        self.in_channel = in_channel
        self.state_initiate = state_initiate
        self.attention_unit = unit
        self.maxout = is_maxout
        self.is_gated_attention = is_gated_attention
        self.is_mutiple = is_mutiple
        self.random_sample = True
        self.greedy = False
        self.y = None
        self.setattr('beam_size')
        self.setattr('attention_unit')

    def set_sublayers(self):
        self.sublayers['state_dec'] = self.core(self.unit_dim, mask=False, in_dim=self.emb_dim)
        self.sublayers['glimpse_dec'] = self.core(self.unit_dim, mask=False, in_dim=self.in_dim)
        self.sublayers['emitter'] = emitter(self.vocab_size, self.in_dim, self.unit_dim, self.emb_dim,
                                            in_dim=self.unit_dim, is_maxout=self.maxout)
        if not self.is_gated_attention:
            self.sublayers['context'] = linear_bias(self.attention_unit, in_dim=self.in_dim)
        else:
            self.sublayers['context'] = linear_bias(self.attention_unit * 2, in_dim=self.in_dim)
        self.sublayers['attention'] = attention(self.attention_unit, self.unit_dim, self.in_dim, self.is_gated_attention,
                                                self.is_mutiple)
        self.sublayers['peek'] = lookuptable(self.emb_dim, in_dim=self.vocab_size)

    def init_params(self):
        self.wt_iniate_s = self.allocate(uniform, 'Wt_iniate_s', weight, self.in_dim // self.in_channel, self.unit_dim)

    def prepare(self, X):
        emb_y = self.sublayers['peek'].feedforward(self.y)
        emb_shifted = T.zeros_like(emb_y)
        emb_shifted = T.set_subtensor(emb_shifted[1:], emb_y[:-1])
        self.y_ = emb_shifted
        state_before_y = self.sublayers['state_dec'].children['input'].feedforward(emb_shifted)
        self.state_before = [state_before_y]
        if self.core != lstm:
            state_before_yg_ = self.sublayers['state_dec'].children['gate'].feedforward(emb_shifted)
            self.state_before.append(state_before_yg_)

        mean_ctx = (self.input[:, :, -(self.in_dim // self.in_channel):] * self.x_mask[:, :, None]).sum(
            0) / self.x_mask.sum(0)[:, None]
        s_0 = mean_ctx
        if self.state_initiate == 'final': s_0 = self.input[0, :, -self.in_dim // self.in_channel:]
        self.s_0 = T.tanh(T.layer_dot(s_0, self.wt_iniate_s))
        self.initiate_state = [self.s_0]
        if self.core == lstm:
            self.initiate_state.append(T.zeros([self.batch_size, self.unit_dim], theano.config.floatX))
        self.initiate_state.append(None)

        self.context = [self.sublayers['context'].feedforward(self.input), self.input, self.x_mask]

        self.n_steps = self.y.shape[0]

    def process_output(self, X):
        self.get_batch_size()
        self.prepare(X)
        if self.core == lstm:
            [self.s, self.c, self.ci], updates = self.apply()
        else:
            [self.s, self.ci], updates = self.apply()
        self.pred = self.sublayers['emitter'].feedforward([self.s, self.y_, self.ci])
        self.output = self.pred.reshape([self.y.shape[0], self.batch_size, self.vocab_size])
        self.updates.update(updates)

    def step(self, *args):
        if self.core == lstm:
            y_, s_, c_, pctx, ctx, x_m = args
            s1, c1 = self.sublayers['state_dec'].step(y_, s_, c_)
        else:
            y_, yg_, s_, pctx, ctx, x_m = args
            s1 = self.sublayers['state_dec'].step(y_, yg_, s_)
        aij = self.sublayers['attention'].feedforward([s1, pctx, x_m])
        if not self.is_mutiple:
            if aij.ndim == 2:
                ci = (ctx * aij[:, :, None]).sum(0)
            else:
                assert aij.ndim == 3
                ci = (ctx * aij[:, :, :, None]).sum(0)
        else:
            ci = (ctx * aij).sum(0)
        if self.core == lstm:
            condition = self.sublayers['glimpse_dec'].children['input'].feedforward(ci)
            s2, c2 = self.sublayers['glimpse_dec'].step(condition, s1, c1)
            s = s2
            return s, c2, ci
        else:
            condition = self.sublayers['glimpse_dec'].children['input'].feedforward(ci)
            conditiong = self.sublayers['glimpse_dec'].children['gate'].feedforward(ci)
            s2 = self.sublayers['glimpse_dec'].step(condition, conditiong, s1)
            s = s2
            return s, ci

    def get_predict(self):
        self.y_0 = T.zeros([self.batch_size, self.emb_dim])
        initiate_state = [self.y_0, self.s_0]
        if self.core == lstm:
            initiate_state.append(T.zeros([self.batch_size, self.unit_dim], theano.config.floatX))
        initiate_state.append(None)

        context = self.context

        def step(*args):
            if self.core == lstm:
                y_emb_, s_, c_, pctx, ctx, x_m = args
                yi = self.sublayers['state_dec'].children['input'].feedforward(y_emb_)
                s, c, ci = self.step(yi, s_, c_, pctx, ctx, x_m)
            else:
                y_emb_, s_, pctx, ctx, x_m = args
                yi = self.sublayers['state_dec'].children['input'].feedforward(y_emb_)
                ygi = self.sublayers['state_dec'].children['gate'].feedforward(y_emb_)
                s, ci = self.step(yi, ygi, s_, pctx, ctx, x_m)

            prob = self.sublayers['emitter'].feedforward([s, y_emb_, ci])
            if self.random_sample:
                prob = self.trng.multinomial(pvals=prob)
            pred = prob.argmax(-1)
            y_emb = self.sublayers['peek'].feedforward(pred).reshape([self.batch_size, self.emb_dim])
            if self.core == lstm:
                return [y_emb, s, c, pred]
            else:
                return [y_emb, s, pred]

        result, updates = theano.apply(step, sequences=[], outputs_info=initiate_state,
                                       non_sequences=context, n_steps=self.y.shape[0])

        self.raw_updates = updates
        self.predict = result[-1]

    def get_sample(self, inputs, beamsize):

        import timeit

        def sample_init():
            x, xm = inputs[0], inputs[2]
            return theano.function([x, xm], [self.s_0, self.context[0], self.context[1]])

        def sample_step(*args):
            if self.core == gru:
                y_i_, s_, ctx, pctx = args
                y_ = self.sublayers['peek'].feedforward(y_i_)
                y_ = T.switch(T.eq(y_i_[:, :, None], 1), T.zeros([self.emb_dim])[None, None, :], y_)
                ys = self.sublayers['state_dec'].children['input'].feedforward(y_)
                ysg = self.sublayers['state_dec'].children['gate'].feedforward(y_)
                s, ci = self.step(ys, ysg, s_, ctx[:, :, None, :], pctx[:, :, None, :], self.x_mask)
                prob = self.sublayers['emitter'].feedforward([s, y_, ci])
                return theano.function([inputs[2], y_i_, s_], [s, prob])
            elif self.core == lstm:
                y_i_, s_, c_, ctx, pctx = args
                y_ = self.sublayers['peek'].feedforward(y_i_)
                y_ = T.switch(T.eq(y_i_[:, :, None], 1), T.zeros([self.emb_dim])[None, None, :], y_)
                ys = self.sublayers['state_dec'].children['input'].feedforward(y_)
                s, c, ci = self.step(ys, s_, c_, ctx, pctx, self.x_mask)
                prob = self.sublayers['emitter'].feedforward([s, y_, ci])
                return theano.function([inputs[2], y_i_, s_, c_], [s, c, prob])

        f_init = sample_init()
        ctx = theano.shared(value=np.zeros([1, 1, self.in_dim]), name='ctx', borrow=True)
        pctx = theano.shared(value=np.zeros([1, 1, self.in_dim]), name='pctx', borrow=True)
        if self.core == lstm:
            ts_ = T.tensor3('s_')
            ty_ = T.imatrix('y_')
            tc_ = T.tensor3('c_')
            f_step = sample_step(ty_, ts_, tc_, ctx, pctx)
        else:
            ts_ = T.tensor3('s_')
            ty_ = T.imatrix('y_')
            f_step = sample_step(ty_, ts_, ctx, pctx)

        def gen_sample(*input):
            maxlen = 50
            s_0, ctx_, pctx_ = f_init(input[0], input[2])
            n_samples = len(s_0)
            masks = np.ones([n_samples, beamsize], 'int8')
            ctx.set_value(ctx_)
            pctx.set_value(pctx_)
            if self.core == lstm:
                y_ = np.ones([n_samples, 1], 'int32')
                s_ = np.tile(s_0[:, None, :], [1, 1, 1])
                c_ = np.zeros([s_0.shape[0], 1, s_0.shape[1]])
            else:
                y_ = np.ones([n_samples, 1], 'int32')
                s_ = np.tile(s_0[:, None, :], [1, 1, 1])
            samples = []
            y_transs = []
            y_mods = []
            probsums = []
            scores = []
            # probs = []
            # states = []
            for _ in range(n_samples):
                samples.append([])
                y_transs.append([])
                y_mods.append([])
                probsums.append(np.zeros([beamsize], 'float32'))
                scores.append([])
            probsums = np.asarray(probsums)
            for n_step in xrange(maxlen):
                if self.core == lstm:
                    c_ = np.asarray(c_, 'float32')
                    s, c, prob_bk_v = f_step(input[2], np.asarray(y_, 'int32'), np.asarray(s_, 'float32'),
                                             np.asarray(c_, 'float32'))
                else:
                    s, prob_bk_v = f_step(input[2], np.asarray(y_, 'int32'), np.asarray(s_, 'float32'))
                # probs.append(prob_bk_v)
                # states.append(s)
                if n_step != 0:
                    prob_bk_v = probsums.flatten()[:, None] - np.log(prob_bk_v * masks.flatten()[:, None])
                    prob_b_kv = prob_bk_v.reshape([n_samples, -1])
                else:
                    prob_bk_v = - np.log(prob_bk_v)
                    prob_b_kv = prob_bk_v.reshape([n_samples, -1])
                if self.core == lstm:
                    s_ = []
                    c_ = []
                    y_ = []
                else:
                    s_ = []
                    y_ = []

                for batch in range(n_samples):
                    mask_b = masks[batch]
                    y_trans_b = y_transs[batch]
                    y_mods_b = y_mods[batch]
                    samples_b = samples[batch]
                    prob = prob_b_kv[batch].flatten()
                    live = mask_b.sum()
                    for i in range(beamsize):
                        if i < live:
                            mask_b[i] = 1
                        else:
                            mask_b[i] = 0

                    y = prob.argpartition(sum(mask_b) - 1)[:sum(mask_b)]
                    y_trans = y // self.vocab_size
                    y_mod = y % self.vocab_size
                    probsums[batch] = np.pad(prob[y], (0, beamsize - mask_b.sum()), 'constant',
                                             constant_values=(0, np.inf))
                    y_trans_b.append(y_trans)
                    y_mods_b.append(y_mod)
                    j = 0
                    if self.core == lstm:
                        y_b = []
                        s_b = []
                        c_b = []
                    else:
                        y_b = []
                        s_b = []
                    for b, m, w in zip(y_trans, y_mod, y):
                        if self.core == lstm:
                            s_b.append(s[batch, b])
                            c_b.append(c[batch, b])
                            y_b.append(m)
                        else:
                            s_b.append(s[batch, b])
                            y_b.append(m)
                        if m == 0:
                            mask_b[j] = 0
                            sample = []
                            l = j
                            for k in range(len(y_mods_b) - 1, -1, -1):
                                sample.append(y_mods_b[k][l])
                                l = y_trans_b[k][l]
                            samples_b.append(sample[::-1])
                            scores[batch].append(prob_b_kv[batch, w] / (n_step + 1))
                        j += 1
                    for j in range(beamsize - len(y)):
                        if self.core == lstm:
                            s_b.append(np.zeros([self.unit_dim]))
                            c_b.append(np.zeros([self.unit_dim]))
                            y_b.append(1)
                        else:
                            s_b.append(np.zeros([self.unit_dim]))
                            y_b.append(1)
                    if self.core == lstm:
                        y_.append(y_b)
                        s_.append(s_b)
                        c_.append(c_b)
                    else:
                        y_.append(y_b)
                        s_.append(s_b)
                    if n_step == maxlen - 1:
                        for o in range(sum(mask_b)):
                            sample = []
                            l = o
                            for k in range(n_step, -1, -1):
                                sample.append(y_mods_b[k][l])
                                l = y_trans_b[k][l]
                            samples_b.append(sample[::-1])
                            scores[batch].append(prob_b_kv[batch, y[o]] / (n_step + 1))
                if (np.equal(masks,0)).all():
                    return [sample[np.array(score).argmin()] for sample,score in zip(samples,scores)]#, samples, scores, probs, states, y_transs, y_mods
            return [sample[np.array(score).argmin()] for sample,score in zip(samples,scores)]#, samples, scores, probs, states, y_transs, y_mods

        return gen_sample

    def gen_sample(self, beam_size=12):
        if beam_size == 1:
            rs = self.random_sample
            self.random_sample = False
            self.greedy = True
            self.get_predict()
            self.greedy = False
            self.random_sample = rs
            return

        y_0 = T.zeros([self.batch_size, beam_size, self.emb_dim])
        y_mm_0 = T.ones([self.batch_size, beam_size], 'int8')
        s_0 = T.tile(self.s_0[:, None, :], [1, beam_size, 1])
        prob_sum_0 = T.zeros([self.batch_size, beam_size])
        initiate_state = [y_0, s_0]
        if self.core == lstm:
            initiate_state.append(T.zeros([self.batch_size, beam_size, self.unit_dim], theano.config.floatX))
        initiate_state.extend([None, None])

        from theano.ifelse import ifelse

        def step(*args):
            if self.core == lstm:
                y_emb_, s_, c_, prob_sum_b_k_, y_mm_b_k_, idx_1_, pctx, ctx, x_m = args
                yi = self.sublayers['state_dec'].children['input'].feedforward(y_emb_)
                s, ci = self.step(yi, s_, c_, pctx, ctx, x_m)
            elif self.core == gru:
                y_emb_, s_, pctx, ctx, x_m = args
                yi = self.sublayers['state_dec'].children['input'].feedforward(y_emb_)
                ygi = self.sublayers['state_dec'].children['gate'].feedforward(y_emb_)
                s, ci = self.step(yi, ygi, s_, pctx[:, :, None, :], ctx[:, :, None, :], x_m)
            else:
                assert False
            prob_bk_v = self.sublayers['emitter'].feedforward([s, y_emb_, ci])
            logprob_b_kv = T.log(prob_bk_v.reshape([self.batch_size, beam_size * self.vocab_size]))
            y_b_k = logprob_b_kv.argsort(-1)[:, :beam_size]

            # prob_sum_bk_v = prob_bk_v  # -T.log(prob_bk_v) + prob_sum_b_k_.flatten()[:, None]  # * y_mm_b_k_.flatten()[:, None]
            # prob_sum_b_kv = ifelse(T.eq(idx_1_, 0), prob_sum_bk_v[::beam_size, :],
            #                     prob_sum_bk_v.reshape([self.n_samples, beam_size * self.vocab_size]))
            # idx_1 = idx_1_ + 1
            # y_b_k = T.argsort(prob_sum_b_kv)[:, :beam_size]

            # y_mod = y_b_k % self.vocab_size
            #  y_trans = y_b_k // self.vocab_size
            # axes = T.repeat(T.arange(self.n_samples), beam_size)
            # s = s[axes, y_trans.flatten(), :].reshape([self.n_samples, beam_size, self.unit_dim])
            # y_emb = self.children['peek'].feedforward(y_mod.flatten()).reshape(
            #     [self.n_samples, beam_size, self.emb_dim])
            #  prob_b_k = prob_sum_b_kv[axes, y_b_k.flatten()].reshape([self.n_samples, beam_size])
            return y_emb_, s, prob_bk_v, y_b_k  # , idx_1, y_mod, y_trans
            '''
            y_mm_c_b_k = T.switch(T.eq(y_mod, 0), 0, 1)
            n_one = T.cast(y_mm_b_k_.sum(-1), 'int32')
            y_mm_tmp = T.arange(beam_size)[None, :]
            y_mm_shifted_b_k = T.switch(y_mm_tmp < n_one[:, None], 1, 0)
            prob_b_k = prob_b_kv[
                T.repeat(T.arange(self.n_samples), beam_size), y_b_k.flatten()].reshape(
                [self.n_samples, beam_size])
            y_mm = y_mm_c_b_k * y_mm_shifted_b_k
            return [y_emb, s, prob_b_k, y_mm, idx_1, y_mm_shifted_b_k, y_trans, y_mod], theano.scan_module.until(
                T.eq(y_mm.sum(), 0))

            '''

        result, updates = theano.apply(step, sequences=[], outputs_info=initiate_state,
                                       non_sequences=self.context, n_steps=50)

        self.result = result

        self.gen_updates = updates

        if self.core == lstm:
            self.y_emb, self.s, self.c, self.prob, self.y_mm, self.idx, self.y_mm_shifted, self.y_trans, self.y_mod = result
        else:
            self.y_emb, self.s, self.prob, self.yy = result

        def dec(mm, ms, t_trans, y_mod, y_prob, y_idx_):
            def d(mm, ms, y_idx_):
                msm = T.xor(mm, ms)
                lc = mm.sum()
                ls = msm.sum()
                eid = T.argsort(msm)[::-1]
                y_idx = T.set_subtensor(y_idx_[lc:lc + ls], eid[:ls])
                score_mark = T.set_subtensor(T.zeros([beam_size])[lc:lc + ls], 1)
                return y_idx, score_mark

            [y_idx_c, score_mark], u = theano.apply(d, [mm, ms, y_idx_])
            ax0 = T.repeat(T.arange(self.batch_size), beam_size)
            y_prob_shift = y_prob[ax0, y_idx_c.flatten()].reshape([self.batch_size, beam_size])
            score = T.switch(T.eq(score_mark, 1), y_prob_shift, T.zeros([beam_size]))
            t_trans = t_trans[ax0, y_idx_c.flatten()].reshape([self.batch_size, beam_size])
            t_mod = y_mod[ax0, y_idx_c.flatten()].reshape([self.batch_size, beam_size])
            y_out = T.switch(T.eq(ms, 1), t_mod, 0)
            y_idx = T.switch(T.eq(ms, 1), t_trans, 0)

            return y_idx, y_out, score

        '''
        idx_list_0 = T.zeros([self.n_samples, beam_size], 'int64')
        [self.y_idx, self.y_out, self.score], dec_updates = theano.scan(dec, sequences=[self.y_mm,
                                                                                        self.y_mm_shifted,
                                                                                        self.y_trans,
                                                                                        self.y_mod,
                                                                                        self.prob],
                                                                        outputs_info=[idx_list_0, None,
                                                                                      None],
                                                                        go_backwards=True)
        self.sample_updates = self.gen_updates
        self.sample_updates.update(dec_updates)
        self.score_sum = (self.score / T.arange(self.score.shape[0], 0, -1, theano.config.floatX)[:, None, None]).sum(0)
        self.choice = self.score_sum.argmin(-1)
        self.samples = self.y_out.dimshuffle(1, 2, 0)[:, :, ::-1]
        self.sample = self.samples[T.arange(self.n_samples), self.choice]
        '''

    def get_cost(self, Y):
        cost = T.nnet.binary_crossentropy(self.pred, Y.flatten())
        cost = cost.reshape([Y.shape[0], Y.shape[1]])
        cost = (cost * self.y_mask).sum(0)
        self.cost = T.mean(cost)

    def get_error(self, Y):
        self.error = T.sum(T.neq(Y, self.predict) * self.y_mask) / T.sum(self.y_mask)


class cnnseq(rechidden):
    def __init__(self, unit, window, core, masked=True, **kwargs):
        super(cnnseq,self).__init__(masked=masked, **kwargs)
        self.unit_dim = unit
        self.core=core
        self.window=window

    def set_sublayers(self):
        self.sublayers['core'] = self.core(self.unit_dim)

    def prepare(self, X):
        self.state_before = []
        self.sublayers['core'].prepare(X)
        core_state_before=self.sublayers['core'].state_before
        for csb in core_state_before:
            self.state_before.append()

        self.initiate_state = [T.zeros([self.batch_size, self.unit_dim], theano.config.floatX)]
        self.context = []
        self.n_steps = self.input.shape[0]

    def step(self, *args):
        x, xg, h_ = args
        gate = xg + T.layer_dot(h_, self.ug)
        gate = T.nnet.sigmoid(gate)
        z_gate = self.slice(gate, 0, self.unit_dim)
        z_gate = self.apply_ops('z_gate', z_gate, dropout, False)
        r_gate = self.slice(gate, 1, self.unit_dim)
        r_gate = self.apply_ops('r_gate', r_gate, dropout, False)
        h_c = T.tanh(x + r_gate * T.layer_dot(h_, self.u))
        h = (1 - z_gate) * h_ + z_gate * h_c
        h = self.apply_ops('hidden_unit', h, dropout)
        h = self.apply_mask(h, args[-2])
        return h

    def apply(self):
        return theano.apply(self.step, sequences=self.state_before, outputs_info=self.initiate_state,
                            non_sequences=self.context, n_steps=self.n_steps, go_backwards=self.go_backwards)
