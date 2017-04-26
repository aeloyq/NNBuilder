# -*- coding: utf-8 -*-
"""
Created on  四月 09 1:22 2017

@author: aeloyq
"""
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict
from nnbuilder.dataprepares import Load_mt
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

lr = 0.5

source_vocab_size = 30000
target_vocab_size = 30000

source_emb_dim = 620
target_emb_dim = 620

enc_dim = 1000
dec_dim = 1000

dec_att_dim = 1000

rng = np.random.RandomState(1234)
trng=RandomStreams(1234)

params = OrderedDict()

X = T.imatrix('X')
Y = T.imatrix('Y')
X_mask = T.matrix('X_Mask')
Y_mask = T.matrix('Y_Mask')


def model():
    ex = params['emb_w'][X.flatten()].reshape([X.shape[0],X.shape[1], source_emb_dim])


    def step_f_enc(e_x, x_m, h_):

        r_gate = T.nnet.sigmoid(T.dot(e_x, params['enc_f_wr']) + T.dot(h_, params['enc_f_ur']))
        z_gate = T.nnet.sigmoid(T.dot(e_x, params['enc_f_wz']) + T.dot(h_, params['enc_f_uz']))

        h_c = T.tanh(T.dot(e_x, params['enc_f_w']) + T.dot(r_gate * h_, params['enc_f_u']))

        h = (1 - z_gate) * h_ + z_gate * h_c

        h = x_m[:, None] * h + (1. - x_m)[:, None] * h_

        return h

    def step_b_enc(e_x, x_m, h_):

        r_gate = T.nnet.sigmoid(T.dot(e_x, params['enc_b_wr']) + T.dot(h_, params['enc_b_ur']))
        z_gate = T.nnet.sigmoid(T.dot(e_x, params['enc_b_wz']) + T.dot(h_, params['enc_b_uz']))

        h_c = T.tanh(T.dot(e_x, params['enc_b_w']) + T.dot(r_gate * h_, params['enc_b_u']))

        h = (1 - z_gate) * h_ + z_gate * h_c

        h = x_m[:, None] * h + (1. - x_m)[:, None] * h_

        return h

    def step_dec(y_t, y_m, e_y, s_, x_m, att_pre,att_in):

        eij = (T.exp(T.dot(T.tanh(T.dot(s_, params['dec_wa']) + att_pre), params['dec_va']))) * x_m
        aij = eij / eij.sum(0, keepdims=True)
        ci = T.sum(aij[:, :, None] * att_in, 0)

        r_gate = T.nnet.sigmoid(
            T.dot(e_y, params['dec_wr']) + T.dot(s_, params['dec_ur']) + T.dot(ci, params['dec_cr']))
        z_gate = T.nnet.sigmoid(
            T.dot(e_y, params['dec_wz']) + T.dot(s_, params['dec_uz']) + T.dot(ci, params['dec_cz']))

        s_c = T.tanh(T.dot(e_y, params['dec_w']) + T.dot(r_gate * s_, params['dec_u']) + T.dot(ci, params['dec_c']))

        s = (1 - z_gate) * s_ + z_gate * s_c

        t_c = T.dot(s_, params['dec_uo']) + T.dot(e_y, params['dec_vo']) + T.dot(ci, params['dec_co'])

        t = T.max(t_c.reshape([t_c.shape[0], 2, t_c.shape[1] // 2]), 1)

        y = T.nnet.softmax(T.dot(t, params['dec_wo']))

        cost = T.nnet.categorical_crossentropy(y, y_t) * y_m

        return s,cost



    def step_dec_sampling(y_t, y_m, e_y, s_, x_m, att_pre,att_in):


        eij = (T.exp(T.dot(T.tanh(T.dot(s_, params['dec_wa']) + att_pre), params['dec_va']))) * x_m
        aij = eij/eij.sum(0,keepdims=True)
        ci = T.sum(aij[:, :, None] * att_in, 0)

        r_gate = T.nnet.sigmoid(
            T.dot(e_y, params['dec_wr']) + T.dot(s_, params['dec_ur']) + T.dot(ci, params['dec_cr']))
        z_gate = T.nnet.sigmoid(
            T.dot(e_y, params['dec_wz']) + T.dot(s_, params['dec_uz']) + T.dot(ci, params['dec_cz']))

        s_c = T.tanh(T.dot(e_y, params['dec_w']) + T.dot(r_gate * s_, params['dec_u']) + T.dot(ci, params['dec_c']))

        s = (1 - z_gate) * s_ + z_gate * s_c

        t_c = T.dot(s_, params['dec_uo']) + T.dot(e_y, params['dec_vo']) + T.dot(ci, params['dec_co'])

        t = T.max(t_c.reshape([t_c.shape[0], 2, t_c.shape[1] // 2]), 1)

        y = T.nnet.softmax(T.dot(t, params['dec_wo']))

        cost = T.nnet.categorical_crossentropy(y, y_t) * y_m

        #y_out=trng.multinomial(pvals=y)

        y_out = T.argmax(y, 1)

        e_y_next = T.reshape(params['dec_e'][y_out], [batch_size, target_emb_dim])

        return e_y_next, s, cost


    batch_size = X.shape[1]

    max_length = X.shape[0]

    out_length = Y.shape[0]

    e_ys = T.reshape(params['dec_e'][Y.flatten()], [out_length, batch_size, target_emb_dim])
    emb_shifted = T.zeros_like(e_ys)
    e_ys = T.set_subtensor(emb_shifted[1:], e_ys[:-1])

    h_0 = T.zeros([batch_size, enc_dim], 'float32')



    o_f_e, u_f_e = theano.scan(fn=step_f_enc, sequences=[ex, X_mask], outputs_info=[h_0], n_steps=max_length)
    o_b_e, u_b_e = theano.scan(fn=step_b_enc, sequences=[ex, X_mask], outputs_info=[h_0], n_steps=max_length,
                               go_backwards=True)

    o_e = T.concatenate([o_f_e, o_b_e[:, ::-1, :]], 2)

    s_0 = T.tanh(T.dot(o_e[0, :, enc_dim:enc_dim * 2], params['dec_ws']))

    y_0 = T.reshape(params['dec_e'][T.zeros([batch_size],'int64')], [batch_size, target_emb_dim])

    atp = T.dot(o_e, params['dec_ua'])

    #o_d, u_d = theano.scan(fn=step_dec_sampling, sequences=[Y, Y_mask], outputs_info=[y_0, s_0,None], non_sequences=[X_mask, atp,o_e],
                          #n_steps=out_length)

    o_d, u_d = theano.scan(fn=step_dec, sequences=[Y, Y_mask,e_ys], outputs_info=[s_0,None], non_sequences=[X_mask, atp,o_e],
                             n_steps=out_length)
    '''
    t_c=o_d[-1]

    t = T.max(t_c.reshape([t_c.shape[0],t_c.shape[1], 2, t_c.shape[2] // 2]), 2)

    y = T.exp(T.dot(t, params['dec_wo']))

    y=y/y.sum(0,keepdims=True)

    y_flat = Y.flatten()
    y_flat_idx = T.arange(y_flat.shape[0]) * target_vocab_size + y_flat
    cost = -T.log(y.flatten()[y_flat_idx])
    cost = cost.reshape([Y.shape[0], Y.shape[1]])
    cost = (cost * Y_mask).sum()'''

    cost=T.mean(o_d[-1])
    model_updates=u_d.items()

    params_updates = [(params[param], params[param] - lr * T.grad(cost, params[param])) for param in params]

    return cost, model_updates+params_updates


def param():
    def alloc(shape, name):
        p = theano.shared(rng.uniform(-1, 1, shape).astype('float32'), name=name, borrow=True)
        params[name] = p
        return p

    emb_w = alloc([source_vocab_size, source_emb_dim], 'emb_w')

    enc_f_w = alloc([source_emb_dim, enc_dim], 'enc_f_w')
    enc_f_wz = alloc([source_emb_dim, enc_dim], 'enc_f_wz')
    enc_f_wr = alloc([source_emb_dim, enc_dim], 'enc_f_wr')
    enc_f_u = alloc([enc_dim, enc_dim], 'enc_f_u')
    enc_f_uz = alloc([enc_dim, enc_dim], 'enc_f_uz')
    enc_f_ur = alloc([enc_dim, enc_dim], 'enc_f_ur')

    enc_b_w = alloc([source_emb_dim, enc_dim], 'enc_b_w')
    enc_b_wz = alloc([source_emb_dim, enc_dim], 'enc_b_wz')
    enc_b_wr = alloc([source_emb_dim, enc_dim], 'enc_b_wr')
    enc_b_u = alloc([enc_dim, enc_dim], 'enc_b_u')
    enc_b_uz = alloc([enc_dim, enc_dim], 'enc_b_uz')
    enc_b_ur = alloc([enc_dim, enc_dim], 'enc_b_ur')

    dec_w = alloc([target_emb_dim, dec_dim], 'dec_w')
    dec_wz = alloc([target_emb_dim, dec_dim], 'dec_wz')
    dec_wr = alloc([target_emb_dim, dec_dim], 'dec_wr')
    dec_u = alloc([dec_dim, dec_dim], 'dec_u')
    dec_uz = alloc([dec_dim, dec_dim], 'dec_uz')
    dec_ur = alloc([dec_dim, dec_dim], 'dec_ur')
    dec_c = alloc([enc_dim * 2, dec_dim], 'dec_c')
    dec_cz = alloc([enc_dim * 2, dec_dim], 'dec_cz')
    dec_cr = alloc([enc_dim * 2, dec_dim], 'dec_cr')
    dec_wa=alloc([dec_dim, dec_att_dim], 'dec_wa')
    dec_ua = alloc([enc_dim*2, dec_att_dim], 'dec_ua')
    dec_va = alloc([dec_att_dim], 'dec_va')
    dec_wo = alloc([dec_dim / 2, target_vocab_size], 'dec_wo')
    dec_uo = alloc([dec_dim, dec_dim], 'dec_uo')
    dec_vo = alloc([target_emb_dim, dec_dim], 'dec_vo')
    dec_co = alloc([enc_dim * 2, dec_dim], 'dec_co')
    dec_e = alloc([target_vocab_size, target_emb_dim], 'dec_e')
    alloc([enc_dim, dec_dim], 'dec_ws')
    '''
    alloc([dec_dim], 'dec_Bis')

    alloc([dec_dim], 'dec_Bir')
    alloc([dec_dim], 'dec_Biz')

    alloc([dec_dim], 'dec_Bi')

    alloc([dec_dim], 'dec_Bio')
    '''

param()

c, u = model()

print 'compile'
f = theano.function([X, Y, X_mask, Y_mask], c, updates=u,on_unused_input='ignore')
print 'compile ok'

import timeit


def t(f, args):
    st = timeit.default_timer()
    f(*args)
    print timeit.default_timer() - st


from nnbuilder import config

config.name = 'mt'
config.data_path = './data/datasets.npz'
config.batch_size = 40
config.valid_batch_size = 64
config.max_epoches = 1000
config.savelog = True
config.transpose_x = True
config.transpose_y = True
config.mask_x = True
config.mask_y = True
config.int_x = True
config.int_y = True
data = Load_mt(maxlen=50, sort_by_len=True,sort_by_asc=False)
import nnbuilder

d = nnbuilder.mainloop.prepare_data(data[0], data[3], range(80))
t(f,d)