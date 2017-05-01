# -*- coding: utf-8 -*-
"""
Created on  三月 19 2:29 2017

@author: aeloyq
"""
'''
trng=config.trng

baselayer_lstm = nnbuilder.layers.lstm.get
baselayer_rnn = nnbuilder.layers.recurrent.get
baselayer_gru = nnbuilder.layers.gru.get
from nnbuilder.layers.basic import utils,baselayer


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
        for key in scan_update:
            u=[(key,scan_update[key])]
            self.updates.extend(u)
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

        t = T.dot(h, self.uo) + T.dot(ci, self.co) + self.bio

        s_0 = T.nnet.softmax(T.dot(t, self.s0wt))
        y = T.nnet.softmax(T.dot(s_0, self.s1wt) + self.s1bi)



        cost=T.nnet.categorical_crossentropy(y,y_true)*y_m

        y = trng.multinomial(pvals=y)

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
        for key in scan_update:
            u=[(key,scan_update[key])]
            self.updates.extend(u)
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
        cost=T.nnet.categorical_crossentropy(y,y_true)
        y = trng.multinomial(pvals=y)
        y_max=T.argmax(y,1)
        r = T.reshape(self.e[y_max] ,[y.shape[0],self.emb_dim])
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

        cost=T.nnet.categorical_crossentropy(y,y_true)
        y = trng.multinomial(pvals=y,dtype='float32')
        r = T.dot(y, self.e)
        return r, h, y,cost

class get_gru_attention_maxout_readout_feedback(baselayer):
    def __init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim,
                 beam_size=12,
                 **kwargs):
        baselayer.__init__(self)
        self.in_dim=in_dim
        self.unit_dim=unit_dim
        self.attention_dim=attention_dim
        self.emb_dim=emb_dim
        self.vocab_dim=vocab_dim
        self.trng=config.trng
        self.beam_size=beam_size

    def init_params(self):
        self.params=[]
        def alloc(shape, name,func):
            p = theano.shared(func(*tuple(shape)), name=name, borrow=True)
            self.params.append(p)
            return p

        self.dec_w = alloc([self.emb_dim, self.unit_dim], 'dec_w', utils.uniform)
        self.dec_wz = alloc([self.emb_dim, self.unit_dim], 'dec_wz', utils.uniform)
        self.dec_wr = alloc([self.emb_dim, self.unit_dim], 'dec_wr', utils.uniform)
        self.dec_u = alloc([self.unit_dim, self.unit_dim], 'dec_u', utils.orthogonal)
        self.dec_uz = alloc([self.unit_dim, self.unit_dim], 'dec_uz', utils.orthogonal)
        self.dec_ur = alloc([self.unit_dim, self.unit_dim], 'dec_ur', utils.orthogonal)
        self.dec_c = alloc([self.in_dim * 2, self.unit_dim], 'dec_c', utils.uniform)
        self.dec_cz = alloc([self.in_dim * 2, self.unit_dim], 'dec_cz', utils.uniform)
        self.dec_cr = alloc([self.in_dim * 2, self.unit_dim], 'dec_cr', utils.uniform)
        self.dec_wa=alloc([self.unit_dim, self.attention_dim], 'dec_wa', utils.uniform)
        self.dec_ua = alloc([self.in_dim*2, self.attention_dim], 'dec_ua', utils.uniform)
        self.dec_va = alloc([self.attention_dim], 'dec_va', utils.uniform)
        self.dec_bia= alloc([self.attention_dim], 'dec_bia', utils.zeros)
        self.dec_wo = alloc([self.unit_dim / 2, self.vocab_dim], 'dec_wo', utils.uniform)
        self.dec_uo = alloc([self.unit_dim, self.unit_dim], 'dec_uo', utils.orthogonal)
        self.dec_vo = alloc([self.emb_dim, self.unit_dim], 'dec_vo', utils.uniform)
        self.dec_co = alloc([self.in_dim * 2, self.unit_dim], 'dec_co', utils.uniform)
        self.dec_e = alloc([self.vocab_dim, self.emb_dim], 'dec_e', utils.uniform)

        self.dec_ws = alloc([self.in_dim, self.unit_dim], 'dec_ws', utils.uniform)
        self.dec_bis = alloc([self.unit_dim], 'dec_Bis', utils.zeros)

        self.dec_bir = alloc([self.unit_dim], 'dec_Bir', utils.zeros)
        self.dec_biz = alloc([self.unit_dim], 'dec_Biz', utils.zeros)

        self.dec_bi = alloc([self.unit_dim], 'dec_Bi', utils.zeros)

        self.dec_bio = alloc([self.unit_dim], 'dec_Bio', utils.zeros)

    def set_y_mask(self, tvar):
        self.y_mask = tvar

    def set_y(self,tvar):
        self.y=tvar

    def set_x_mask(self, tvar):
        self.x_mask = tvar

    def get_output(self):
        batch_size = self.y.shape[1]
        out_length = self.y.shape[0]
        s_0 = T.tanh(T.dot(self.input[0,:,self.in_dim:self.in_dim*2], self.dec_ws) + self.dec_bis)
        atp=T.dot(self.input,self.dec_ua)+self.dec_bia
        e_ys = T.reshape(self.dec_e[self.y.flatten()], [out_length, batch_size, self.emb_dim])
        emb_shifted = T.zeros_like(e_ys)
        e_ys = T.set_subtensor(emb_shifted[1:], e_ys[:-1])

        def step(y_t, y_m, e_y, s_, x_m, att_pre,att_in):

            eij = (T.exp(T.dot(T.tanh(T.dot(s_, self.dec_wa) + att_pre), self.dec_va))) * x_m
            aij = eij / eij.sum(0, keepdims=True)
            ci = T.sum(aij[:, :, None] * att_in, 0)

            r_gate = T.nnet.sigmoid(
                T.dot(e_y,self.dec_wr) + T.dot(s_, self.dec_ur) + T.dot(ci, self.dec_cr)+self.dec_bir)
            z_gate = T.nnet.sigmoid(
                T.dot(e_y, self.dec_wz) + T.dot(s_, self.dec_uz) + T.dot(ci, self.dec_cz)+self.dec_biz)

            s_c = T.tanh(T.dot(e_y, self.dec_w)+ T.dot(r_gate * s_, self.dec_u) + T.dot(ci, self.dec_c)+self.dec_bi)

            s = (1 - z_gate) * s_ + z_gate * s_c

            t_c = T.dot(s_, self.dec_uo) +T.dot(e_y, self.dec_vo)+ T.dot(ci, self.dec_co)+self.dec_bio

            t = T.max(t_c.reshape([t_c.shape[0], 2, t_c.shape[1] // 2]), 1)

            y = T.nnet.softmax(T.dot(t, self.dec_wo))

            cost = T.nnet.categorical_crossentropy(y, y_t) * y_m

            return s, cost,y
        def step_beamsearch(e_y_list, s_list_, att_pre,att_in):

            def step_inner (e_y, s_, att_pre,att_in):

                eij=(T.dot(T.tanh(T.dot(s_,self.dec_wa)+att_pre),self.dec_va))
                aij=T.nnet.softmax(eij.dimshuffle(1,0)).dimshuffle(1,0)
                ci=T.sum(aij[:,:,None]*att_in,0)

                r_gate = T.nnet.sigmoid(
                    T.dot(e_y, self.dec_wr) + T.dot(s_, self.dec_ur) + T.dot(ci, self.dec_cr)+self.dec_bir)
                z_gate = T.nnet.sigmoid(
                    T.dot(e_y, self.dec_wz) + T.dot(s_, self.dec_uz) + T.dot(ci, self.dec_cz)+self.dec_biz)

                s_c = T.tanh(T.dot(e_y, self.dec_w) + T.dot(r_gate * s_, self.dec_u) + T.dot(ci, self.dec_c)+self.dec_bi)

                s = (1 - z_gate) * s_ + z_gate * s_c

                t_c = T.dot(s_, self.dec_uo) + T.dot(e_y, self.dec_vo) + T.dot(ci, self.dec_co)+self.dec_bio

                t = T.max(t_c.reshape([t_c.shape[0], 2, t_c.shape[1] // 2]), 1)

                y = T.nnet.softmax(T.dot(t, self.dec_wo))

                y_out = T.argsort(y,1)[:,self.beam_size:]

                e_y_next = T.reshape(self.dec_e[y_out.flatten()], [batch_size, self.emb_dim])

                return e_y_next, s, y_out

            [e_y_list_next, s_list, y_out],u=theano.scan(step_inner,sequences=[e_y_list,s_list_],outputs_info=[],non_sequences=[att_pre,att_in],n_steps=self.x_mask.shape[0]*2)

            return e_y_list_next, s_list,y_out
        [s,cost,y], scan_update = theano.scan(step,
                                           sequences=[self.y,self.y_mask,e_ys],
                                           outputs_info=[s_0, None,None],
                                           non_sequences=[self.x_mask,atp,self.input],
                                           name=self.name + '_Scan',
                                           n_steps=out_length)
        self.output = y
        self.loss=T.mean(cost)
        self.updates=scan_update.items()
        self.predict()



    def predict(self):

        self.pred_Y=self.output.argmax(2)

    def cost(self, Y):
        return self.loss

    def error(self,Y):
        return T.mean(T.neq(self.pred_Y,self.y)*self.y_mask)

class get_gru_attention_maxout_readout_feedback_g(baselayer):
    def __init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim,
                 beam_size=12,
                 **kwargs):
        baselayer.__init__(self)
        self.in_dim=in_dim
        self.unit_dim=unit_dim
        self.attention_dim=attention_dim
        self.emb_dim=emb_dim
        self.vocab_dim=vocab_dim
        self.trng=config.trng
        self.beam_size=beam_size

    def init_params(self):
        self.params=[]
        def alloc(shape, name,func):
            p = theano.shared(func(*tuple(shape)), name=name, borrow=True)
            self.params.append(p)
            return p

        self.dec_w = alloc([self.emb_dim, self.unit_dim], 'dec_w', utils.uniform)
        self.dec_wz = alloc([self.emb_dim, self.unit_dim], 'dec_wz', utils.uniform)
        self.dec_wr = alloc([self.emb_dim, self.unit_dim], 'dec_wr', utils.uniform)
        self.dec_u = alloc([self.unit_dim, self.unit_dim], 'dec_u', utils.orthogonal)
        self.dec_uz = alloc([self.unit_dim, self.unit_dim], 'dec_uz', utils.orthogonal)
        self.dec_ur = alloc([self.unit_dim, self.unit_dim], 'dec_ur', utils.orthogonal)
        self.dec_c = alloc([self.in_dim * 2, self.unit_dim], 'dec_c', utils.uniform)
        self.dec_cz = alloc([self.in_dim * 2, self.unit_dim], 'dec_cz', utils.uniform)
        self.dec_cr = alloc([self.in_dim * 2, self.unit_dim], 'dec_cr', utils.uniform)
        self.dec_wa=alloc([self.unit_dim, self.attention_dim], 'dec_wa', utils.uniform)
        self.dec_ua = alloc([self.in_dim*2, self.attention_dim], 'dec_ua', utils.uniform)
        self.dec_va = alloc([self.attention_dim,self.in_dim*2], 'dec_va', utils.uniform)
        self.dec_bia= alloc([self.attention_dim], 'dec_bia', utils.zeros)
        self.dec_wo = alloc([self.unit_dim / 2, self.vocab_dim], 'dec_wo', utils.uniform)
        self.dec_uo = alloc([self.unit_dim, self.unit_dim], 'dec_uo', utils.orthogonal)
        self.dec_vo = alloc([self.emb_dim, self.unit_dim], 'dec_vo', utils.uniform)
        self.dec_co = alloc([self.in_dim * 2, self.unit_dim], 'dec_co', utils.uniform)
        self.dec_e = alloc([self.vocab_dim, self.emb_dim], 'dec_e', utils.uniform)

        self.dec_ws = alloc([self.in_dim*2, self.unit_dim], 'dec_ws', utils.uniform)
        self.dec_bis = alloc([self.unit_dim], 'dec_Bis', utils.zeros)

        self.dec_bir = alloc([self.unit_dim], 'dec_Bir', utils.zeros)
        self.dec_biz = alloc([self.unit_dim], 'dec_Biz', utils.zeros)

        self.dec_bi = alloc([self.unit_dim], 'dec_Bi', utils.zeros)

        self.dec_bio = alloc([self.unit_dim], 'dec_Bio', utils.zeros)

    def set_y_mask(self, tvar):
        self.y_mask = tvar

    def set_y(self,tvar):
        self.y=tvar

    def set_x_mask(self, tvar):
        self.x_mask = tvar

    def get_output(self):
        batch_size = self.y.shape[1]
        out_length = self.y.shape[0]
        s_0 = T.tanh(T.dot(self.input, self.dec_ws).sum(0) + self.dec_bis)
        e_ys = T.reshape(self.dec_e[self.y.flatten()], [out_length, batch_size, self.emb_dim])
        emb_shifted = T.zeros_like(e_ys)
        e_ys = T.set_subtensor(emb_shifted[1:], e_ys[:-1])
        atp=T.dot(self.input,self.dec_ua)+self.dec_bia

        def step(y_t, y_m, e_y, s_, x_m, att_pre,att_in):

            eij=(T.exp(T.dot(T.tanh(T.dot(s_,self.dec_wa)+att_pre),self.dec_va)))*x_m
            aij=eij/eij.sum(0,keepdims=True)
            ci=(aij*att_in).sum(0)

            r_gate = T.nnet.sigmoid(
                T.dot(e_y, self.dec_wr) + T.dot(s_, self.dec_ur) + T.dot(ci, self.dec_cr)+self.dec_bir)
            z_gate = T.nnet.sigmoid(
                T.dot(e_y, self.dec_wz) + T.dot(s_, self.dec_uz) + T.dot(ci, self.dec_cz)+self.dec_biz)

            s_c = T.tanh(T.dot(e_y, self.dec_w) + T.dot(r_gate * s_, self.dec_u) + T.dot(ci, self.dec_c)+self.dec_bi)

            s = (1 - z_gate) * s_ + z_gate * s_c

            t_c = T.dot(s_, self.dec_uo) + T.dot(e_y, self.dec_vo) + T.dot(ci, self.dec_co)+self.dec_bio

            t = T.max(t_c.reshape([t_c.shape[0], 2, t_c.shape[1] // 2]), 1)

            y = T.nnet.softmax(T.dot(t, self.dec_wo))

            cost = T.nnet.categorical_crossentropy(y, y_t) * y_m

            return  s, cost,y
        def step_beamsearch(y_t, y_m, e_y, s_, x_m, att_pre,att_in):

            eij=(T.dot(T.tanh(T.dot(s_,self.dec_wa)+att_pre),self.dec_va))*x_m
            aij=T.nnet.softmax(eij.dimshuffle(1,0)).dimshuffle(1,0)
            ci=T.sum(aij[:,:,None]*att_in,0)

            r_gate = T.nnet.sigmoid(
                T.dot(e_y, self.dec_wr) + T.dot(s_, self.dec_ur) + T.dot(ci, self.dec_cr)+self.dec_bir)
            z_gate = T.nnet.sigmoid(
                T.dot(e_y, self.dec_wz) + T.dot(s_, self.dec_uz) + T.dot(ci, self.dec_cz)+self.dec_biz)

            s_c = T.tanh(T.dot(e_y, self.dec_w) + T.dot(r_gate * s_, self.dec_u) + T.dot(ci, self.dec_c)+self.dec_bi)

            s = (1 - z_gate) * s_ + z_gate * s_c

            t_c = T.dot(s_, self.dec_uo) + T.dot(e_y, self.dec_vo) + T.dot(ci, self.dec_co)+self.dec_bio

            t = T.max(t_c.reshape([t_c.shape[0], 2, t_c.shape[1] // 2]), 1)

            y = T.nnet.softmax(T.dot(t, self.dec_wo))

            cost = T.nnet.categorical_crossentropy(y, y_t) * y_m

            y_rng=trng.multinomial(pvals=y)

            y_out = T.argmax(y_rng, 1)

            e_y_next = T.reshape(self.dec_e[y_out], [batch_size, self.emb_dim])

            return e_y_next, s, cost,y_out

        [s, cost, y], scan_update = theano.scan(step,
                                                sequences=[self.y, self.y_mask, e_ys],
                                                outputs_info=[s_0, None, None],
                                                non_sequences=[self.x_mask, atp, self.input],
                                                name=self.name + '_Scan',
                                                n_steps=out_length)
        self.output = y
        self.loss = T.mean(cost)
        self.updates = scan_update.items()
        self.predict()

    def predict(self):
        self.pred_Y = self.output.argmax(2)

    def cost(self, Y):
        return self.loss

    def error(self,Y):
        return T.mean(T.neq(self.pred_Y,self.y)*self.y_mask)

class get_gru_attention_maxout_readout_feedback_ug(baselayer):
    def __init__(self, in_dim, unit_dim, attention_dim, emb_dim, vocab_dim,
                 beam_size=12,
                 **kwargs):
        baselayer.__init__(self)
        self.in_dim=in_dim
        self.unit_dim=unit_dim
        self.attention_dim=attention_dim
        self.emb_dim=emb_dim
        self.vocab_dim=vocab_dim
        self.trng=config.trng
        self.beam_size=beam_size

    def init_params(self):
        self.params=[]
        def alloc(shape, name,func):
            p = theano.shared(func(*tuple(shape)), name=name, borrow=True)
            self.params.append(p)
            return p

        self.dec_w = alloc([self.emb_dim, self.unit_dim], 'dec_w', utils.uniform)
        self.dec_wz = alloc([self.emb_dim, self.unit_dim], 'dec_wz', utils.uniform)
        self.dec_wr = alloc([self.emb_dim, self.unit_dim], 'dec_wr', utils.uniform)
        self.dec_u = alloc([self.unit_dim, self.unit_dim], 'dec_u', utils.orthogonal)
        self.dec_uz = alloc([self.unit_dim, self.unit_dim], 'dec_uz', utils.orthogonal)
        self.dec_ur = alloc([self.unit_dim, self.unit_dim], 'dec_ur', utils.orthogonal)
        self.dec_c = alloc([self.in_dim * 2, self.unit_dim], 'dec_c', utils.uniform)
        self.dec_cz = alloc([self.in_dim * 2, self.unit_dim], 'dec_cz', utils.uniform)
        self.dec_cr = alloc([self.in_dim * 2, self.unit_dim], 'dec_cr', utils.uniform)
        self.dec_wa=alloc([self.unit_dim, self.attention_dim], 'dec_wa', utils.uniform)
        self.dec_ua = alloc([self.in_dim*2, self.attention_dim], 'dec_ua', utils.uniform)
        self.dec_va = alloc([self.attention_dim], 'dec_va', utils.uniform)
        self.dec_bia= alloc([self.attention_dim], 'dec_bia', utils.zeros)
        self.dec_wo = alloc([self.unit_dim / 2, self.vocab_dim], 'dec_wo', utils.uniform)
        self.dec_uo = alloc([self.unit_dim, self.unit_dim], 'dec_uo', utils.orthogonal)
        self.dec_vo = alloc([self.emb_dim, self.unit_dim], 'dec_vo', utils.uniform)
        self.dec_co = alloc([self.in_dim * 2, self.unit_dim], 'dec_co', utils.uniform)
        self.dec_e = alloc([self.vocab_dim, self.emb_dim], 'dec_e', utils.uniform)

        self.dec_ws = alloc([self.in_dim, self.unit_dim], 'dec_ws', utils.uniform)
        self.dec_bis = alloc([self.unit_dim], 'dec_Bis', utils.zeros)

        self.dec_bir = alloc([self.unit_dim], 'dec_Bir', utils.zeros)
        self.dec_biz = alloc([self.unit_dim], 'dec_Biz', utils.zeros)

        self.dec_bi = alloc([self.unit_dim], 'dec_Bi', utils.zeros)

        self.dec_bio = alloc([self.unit_dim], 'dec_Bio', utils.zeros)

    def set_y_mask(self, tvar):
        self.y_mask = tvar

    def set_y(self,tvar):
        self.y=tvar

    def set_x_mask(self, tvar):
        self.x_mask = tvar

    def get_output(self):
        batch_size = self.y.shape[1]
        out_length = self.y.shape[0]
        s_0 = T.tanh(T.dot(self.input[0,:,self.in_dim:self.in_dim*2], self.dec_ws) + self.dec_bis)
        y_0 = self.dec_e[T.zeros([batch_size],'int64')].reshape([batch_size,self.emb_dim])
        atp=T.dot(self.input,self.dec_ua)+self.dec_bia

        def step(y_t, y_m, e_y, s_, x_m, att_pre,att_in):

            eij=(T.dot(T.tanh(T.dot(s_,self.dec_wa)+att_pre),self.dec_va))*x_m
            aij=T.nnet.softmax(eij.dimshuffle(1,0)).dimshuffle(1,0)
            ci=T.sum(aij[:,:,None]*att_in,0)

            r_gate = T.nnet.sigmoid(
                T.dot(e_y, self.dec_wr) + T.dot(s_, self.dec_ur) + T.dot(ci, self.dec_cr)+self.dec_bir)
            z_gate = T.nnet.sigmoid(
                T.dot(e_y, self.dec_wz) + T.dot(s_, self.dec_uz) + T.dot(ci, self.dec_cz)+self.dec_biz)

            s_c = T.tanh(T.dot(e_y, self.dec_w) + T.dot(r_gate * s_, self.dec_u) + T.dot(ci, self.dec_c)+self.dec_bi)

            s = (1 - z_gate) * s_ + z_gate * s_c

            t_c = T.dot(s_, self.dec_uo) + T.dot(e_y, self.dec_vo) + T.dot(ci, self.dec_co)+self.dec_bio

            t = T.max(t_c.reshape([t_c.shape[0], 2, t_c.shape[1] // 2]), 1)

            y = T.nnet.softmax(T.dot(t, self.dec_wo))

            cost = T.nnet.categorical_crossentropy(y, y_t) * y_m

            y_rng=trng.multinomial(pvals=y)

            y_out = T.argmax(y_rng, 1)

            e_y_next = T.reshape(self.dec_e[y_out], [batch_size, self.emb_dim])

            return e_y_next, s, cost,y_out
        def step_beamsearch(y_t, y_m, e_y, s_, x_m, att_pre,att_in):

            eij=(T.dot(T.tanh(T.dot(s_,self.dec_wa)+att_pre),self.dec_va))*x_m
            aij=T.nnet.softmax(eij.dimshuffle(1,0)).dimshuffle(1,0)
            ci=T.sum(aij[:,:,None]*att_in,0)

            r_gate = T.nnet.sigmoid(
                T.dot(e_y, self.dec_wr) + T.dot(s_, self.dec_ur) + T.dot(ci, self.dec_cr)+self.dec_bir)
            z_gate = T.nnet.sigmoid(
                T.dot(e_y, self.dec_wz) + T.dot(s_, self.dec_uz) + T.dot(ci, self.dec_cz)+self.dec_biz)

            s_c = T.tanh(T.dot(e_y, self.dec_w) + T.dot(r_gate * s_, self.dec_u) + T.dot(ci, self.dec_c)+self.dec_bi)

            s = (1 - z_gate) * s_ + z_gate * s_c

            t_c = T.dot(s_, self.dec_uo) + T.dot(e_y, self.dec_vo) + T.dot(ci, self.dec_co)+self.dec_bio

            t = T.max(t_c.reshape([t_c.shape[0], 2, t_c.shape[1] // 2]), 1)

            y = T.nnet.softmax(T.dot(t, self.dec_wo))

            cost = T.nnet.categorical_crossentropy(y, y_t) * y_m

            y_rng=trng.multinomial(pvals=y)

            y_out = T.argmax(y_rng, 1)

            e_y_next = T.reshape(self.dec_e[y_out], [batch_size, self.emb_dim])

            return e_y_next, s, cost,y_out
        [e_y,s,cost,y_out], scan_update = theano.scan(step,
                                           sequences=[self.y,self.y_mask],
                                           outputs_info=[y_0, s_0, None,None],
                                           non_sequences=[self.x_mask,atp,self.input],
                                           name=self.name + '_Scan',
                                           n_steps=out_length)
        self.debug_stream.extend([s])
        self.output = y_out
        self.loss=T.mean(cost)
        self.updates=scan_update.items()
        self.predict()



    def predict(self):

        self.pred_Y=self.output

    def cost(self, Y):
        return self.loss

    def error(self,Y):
        return T.mean(T.neq(self.pred_Y,self.y)*self.y_mask)

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
'''