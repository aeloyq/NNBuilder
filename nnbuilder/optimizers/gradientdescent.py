# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:05 PM 2017

@author: aeloyq
"""

import numpy as np
from nnbuilder.kernel import *
from collections import OrderedDict


class Sgd(object):
    def __init__(self):
        self.params = None
        self.configuration = {}
        self.loss = None
        self.learning_rate = 0.01
        self.clip = False
        self.clip_norm = 1.

    def init(self, wrt, loss):
        self.params = wrt
        self.loss = loss
        if not isinstance(self.learning_rate,kernel.shared):
            self.learning_rate = kernel.shared(self.numpy_floatX(self.learning_rate), name='Learning_Rate',attr=[None])
        self.gparams = OrderedDict()
        self.updates2params = OrderedDict()
        self.updates = OrderedDict()

    def get_grad(self):
        self.iter_dict(lambda x: kernel.grad(self.loss, x), self.params, self.gparams)
        if self.clip:
            self.iter_dict(lambda x: self.grad_clip(x), self.gparams, self.gparams)

    def get_updates(self):
        self.get_grad()
        self.iter_dict(lambda x: self.learning_rate * x, self.gparams, self.updates2params)
        self.iter_updates()
        return self.updates

    def grad_clip(self, grad):
        return T.clip(grad, -self.clip_norm, self.clip_norm)

    def numpy_floatX(self, data):
        return np.asarray(data, dtype=kernel.config.floatX)

    def iter_dict(self, fn, dict1, dict2):
        for name, elem in dict1.items():
            dict2[name] = fn(elem)

    def iter_dict_(self, fn, dict1, dict2, dict3):
        for name, elem1 in dict1.items():
            elem2 = dict2[name]
            dict3[name] = fn(elem1, elem2)

    def iter_dict__(self, fn, dict1, dict2, dict3, dict4):
        for name, elem1 in dict1.items():
            elem2 = dict2[name]
            elem3 = dict3[name]
            dict4[name] = fn(elem1, elem2, elem3)

    def iter_updates(self):
        for name, delta in self.updates2params.items():
            self.updates[self.params[name]] = (self.params[name] - delta)

    def iter_register(self, shared_data, current_data, target_dict):
        for name, cd in current_data.items():
            target_dict[shared_data[name]] = cd

    def iter_accumulator(self, shared_data, current_data, target_dict):
        for name, cd in current_data.items():
            target_dict[shared_data[name]] = (shared_data[name] + cd)

    def iter_mutiplication(self, shared_data, current_data, target_dict):
        for name, cd in current_data.items():
            target_dict[shared_data[name]] = (shared_data[name] * cd)

    def save_data(self, name, data, dict):
        dict[name] = OrderedDict()
        for d in data.values():
            dict[name][d.name] = d.get()

    def load_data(self, name, data, dict):
        for d in data.values():
            d.set(dict[name][d.name])

    def save_(self, dict):
        pass

    def load_(self, dict):
        pass

    def set(self, **kwargs):
        for key, val in kwargs.items():
            self.__setattr__(key, val)


sgd = Sgd()


class Momentum(Sgd):
    def __init__(self):
        super(Momentum, self).__init__()
        self.learning_rate = 0.01
        self.momentum_factor = 0.9

    def init(self, wt_packs, loss):
        super(Momentum, self).init(wt_packs, loss)
        self.mf = kernel.shared(self.numpy_floatX(self.momentum_factor)
                                , name='momentum_factor',attr=[None])
        self.pu = OrderedDict()
        self.iter_dict(lambda x: kernel.shared(x.get() * self.numpy_floatX(0.),
                                               name='momentum_pre_update_%s' % x.name,attr=x.attr), self.params,
                       self.pu)
        self.updates_pu = OrderedDict()

    def get_updates(self):
        self.get_grad()
        self.iter_dict_(lambda x, y: self.learning_rate * (self.mf * y + x), self.gparams,
                        self.pu, self.updates2params)
        self.iter_updates()
        self.iter_register(self.pu, self.updates2params, self.updates_pu)
        self.updates.update(self.updates_pu)
        return self.updates

    def save_(self, dict):
        self.save_data('pu', self.pu, dict)
        return dict

    def load_(self, dict):
        self.load_data('pu', self.pu, dict)


momentum = Momentum()


class Nag(Sgd):
    def __init__(self):
        super(Nag, self).__init__()
        self.learning_rate = 0.01
        self.beta = 0.9

    def init(self, wt_packs, loss):
        super(Nag, self).init(wt_packs, loss)
        self.beta = kernel.shared(self.numpy_floatX(self.beta)
                                  , name='momentum_factor',attr=[None])
        self.pg = OrderedDict()
        self.iter_dict(lambda x: kernel.shared(x.get() * self.numpy_floatX(0.),
                                               name='momentum_pre_grad_%s' % x.name,attr=x.attr), self.params,
                       self.pg)
        self.pu = OrderedDict()
        self.iter_dict(lambda x: kernel.shared(x.get() * self.numpy_floatX(0.),
                                               name='momentum_pre_update_%s' % x.name,attr=x.attr), self.params,
                       self.pu)
        self.updates_pg = OrderedDict()
        self.updates_pu = OrderedDict()

    def get_updates(self):
        self.get_grad()
        self.iter_dict__(lambda x, y, z: self.learning_rate * (self.beta * z + x + self.beta * (x - y)), self.gparams,
                         self.pg, self.pu, self.updates2params)
        self.iter_updates()
        self.iter_register(self.pg, self.gparams, self.updates_pg)
        self.iter_register(self.pu, self.updates2params, self.updates_pu)
        self.updates.update(self.updates_pg)
        self.updates.update(self.updates_pu)
        return self.updates

    def save_(self, dict):
        self.save_data('pg', self.pg, dict)
        self.save_data('pu', self.pu, dict)
        return dict

    def load_(self, dict):
        self.load_data('pg', self.pg, dict)
        self.load_data('pu', self.pu, dict)


nag = Nag()


class Adagrad(Sgd):
    def __init__(self):
        super(Adagrad, self).__init__()
        self.learning_rate = 0.01
        self.epsilon = 1e-8

    def init(self, wt_packs, loss):
        super(Adagrad, self).init(wt_packs, loss)
        self.epsilon = kernel.shared(self.numpy_floatX(self.epsilon)
                                     , name='epsilon',attr=[None])
        self.au = OrderedDict()
        self.iter_dict(lambda x: kernel.shared(x.get() * self.numpy_floatX(0.),
                                               name='adagrad_accumulated_updates_%s' % x.name,attr=x.attr),
                       self.params,
                       self.au)
        self.updates_au = OrderedDict()

    def get_updates(self):
        self.get_grad()
        square_gradient = OrderedDict()
        self.iter_dict(lambda x: x ** 2, self.gparams, square_gradient)
        self.iter_dict__(lambda x, y, z: (self.learning_rate / (T.sqrt(z + y) + self.epsilon)) * x, self.gparams,
                         square_gradient, self.au, self.updates2params)
        self.iter_updates()
        self.iter_accumulator(self.au, square_gradient, self.updates_au)
        self.updates.update(self.updates_au)
        return self.updates

    def save_(self, dict):
        self.save_data('au', self.au, dict)
        return dict

    def load_(self, dict):
        self.load_data('au', self.au, dict)


adagrad = Adagrad()


class Rmsprop(Sgd):
    def __init__(self):
        super(Rmsprop, self).__init__()
        self.rou = 0.5
        self.epsilon = 1e-8

    def init(self, wt_packs, loss):
        super(Rmsprop, self).init(wt_packs, loss)
        self.rou = kernel.shared(self.numpy_floatX(self.rou)
                                 , name='rou',attr=[None])
        self.epsilon = kernel.shared(self.numpy_floatX(self.epsilon)
                                     , name='epsilon',attr=[None])
        self.pEg2 = OrderedDict()
        self.updates_Eg2 = OrderedDict()
        self.iter_dict(lambda x: kernel.shared(x.get() * self.numpy_floatX(0.),
                                               name='rmsprop_pre_grad2_%s' % x.name,attr=x.attr), self.params,
                       self.pEg2)

    def get_updates(self):
        self.get_grad()
        cEg2 = OrderedDict()
        self.iter_dict_(lambda x, y: self.rou * y + (1 - self.rou) * (x ** 2), self.gparams, self.pEg2, cEg2)
        self.iter_dict_(lambda x, y, z: (self.learning_rate / T.sqrt(y + self.epsilon)) * x, self.gparams,
                        cEg2, self.updates2params)
        self.iter_updates()
        self.iter_register(self.pEg2, cEg2, self.updates_Eg2)
        self.updates.update(self.updates_Eg2)
        return self.updates

    def save_(self, dict):
        self.save_data('Eg2', self.pEg2, dict)
        return dict

    def load_(self, dict):
        self.load_data('Eg2', self.pEg2, dict)


rmsprop = Rmsprop


class Adadelta(Sgd):
    def __init__(self):
        super(Adadelta, self).__init__()
        self.rou = 0.95
        self.epsilon = 1e-6

    def init(self, wt_packs, loss):
        super(Adadelta, self).init(wt_packs, loss)
        self.rou = kernel.shared(self.numpy_floatX(self.rou)
                                 , name='rou',attr=[None])
        self.epsilon = kernel.shared(self.numpy_floatX(self.epsilon)
                                     , name='epsilon',attr=[None])
        self.pEg2 = OrderedDict()
        self.pEu2 = OrderedDict()
        self.train_updates = OrderedDict()
        self.updates_Eg2 = OrderedDict()
        self.updates_Eu2 = OrderedDict()
        self.iter_dict(lambda x: kernel.shared(x.get() * self.numpy_floatX(0.),
                                               name='adadelta_pre_grad2_%s' % x.name,attr=x.attr), self.params,
                       self.pEg2)
        self.iter_dict(lambda x: kernel.shared(x.get() * self.numpy_floatX(0.),
                                               name='adadelta_pre_update2_%s' % x.name,attr=x.attr), self.params,
                       self.pEu2)
        self.updates = OrderedDict()

    def get_updates(self):
        self.get_grad()
        cEg2 = OrderedDict()
        cEu2 = OrderedDict()
        self.iter_dict_(lambda x, y: self.rou * y + (1 - self.rou) * (x ** 2), self.gparams, self.pEg2, cEg2)
        self.iter_dict__(lambda x, y, z: (T.sqrt(z + self.epsilon) / T.sqrt(y + self.epsilon)) * x, self.gparams,
                         cEg2, self.pEu2, self.updates2params)
        self.iter_dict_(lambda x, y: self.rou * y + (1 - self.rou) * (x ** 2), self.updates2params, self.pEu2,
                        cEu2)
        self.iter_updates()
        self.iter_register(self.pEg2, cEg2, self.updates_Eg2)
        self.iter_register(self.pEu2, cEu2, self.updates_Eu2)
        self.updates.update(self.updates_Eg2)
        self.updates.update(self.updates_Eu2)
        return self.updates

    def save_(self, dict):
        self.save_data('Eg2', self.pEg2, dict)
        self.save_data('Edx2', self.pEu2, dict)
        return dict

    def load_(self, dict):
        self.load_data('Eg2', self.pEg2, dict)
        self.load_data('Edx2', self.pEu2, dict)


adadelta = Adadelta()


class Radam(Sgd):
    def __init__(self):
        super(Radam, self).__init__()
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.epsilon = 1e-8

    def init(self, wrt, loss):
        super(Radam, self).init(wrt, loss)
        self.beta_1_ = kernel.shared(self.numpy_floatX(self.beta_1)
                                     , name='beta_1',attr=[None])
        self.beta_2_ = kernel.shared(self.numpy_floatX(self.beta_2)
                                     , name='beta_2',attr=[None])
        self.epsilon_ = kernel.shared(self.numpy_floatX(self.epsilon)
                                      , name='epsilon',attr=[None])
        self.t_ = kernel.shared(self.numpy_floatX(1)
                                , name='t',attr=[None])
        self.pm = OrderedDict()
        self.pv = OrderedDict()
        self.updates_m = OrderedDict()
        self.updates_v = OrderedDict()
        self.iter_dict(lambda x: kernel.shared(x.get() * self.numpy_floatX(0.),
                                               name='adam_pre_m_%s' % x.name,attr=x.attr), self.params,
                       self.pm)
        self.iter_dict(lambda x: kernel.shared(x.get() * self.numpy_floatX(0.),
                                               name='adam_pre_v_%s' % x.name,attr=x.attr),
                       self.params,
                       self.pv)
        self.updates = OrderedDict()

    def get_updates(self):
        self.get_grad()
        cm = OrderedDict()
        cv = OrderedDict()
        cm_e = OrderedDict()
        cv_e = OrderedDict()
        self.iter_dict_(lambda x, y: self.beta_1_ * y + (1 - self.beta_1_) * x, self.gparams,
                        self.pm, cm)
        self.iter_dict_(lambda x, y: self.beta_2_ * y + (1 - self.beta_2_) * (x ** 2), self.gparams,
                        self.pv, cv)
        self.iter_dict(lambda x: x / (1 - self.beta_1_ ** self.t_), cm, cm_e)
        self.iter_dict(lambda x: x / (1 - self.beta_2_ ** self.t_), cv, cv_e)
        self.iter_dict_(lambda x, y: self.learning_rate * (x / (T.sqrt(y) + self.epsilon_)), cm_e,
                        cv_e, self.updates2params)
        self.iter_updates()
        self.iter_register(self.pm, cm, self.updates_m)
        self.iter_register(self.pv, cv, self.updates_v)
        self.updates.update(self.updates_m)
        self.updates.update(self.updates_v)
        self.updates.update({self.t_: self.t_ + 1})
        return self.updates

    def save_(self, dict):
        self.save_data('pm', self.pm, dict)
        self.save_data('pv', self.pv, dict)
        dict['t'] = self.t_.get()
        return dict

    def load_(self, dict):
        self.load_data('pm', self.pm, dict)
        self.load_data('pv', self.pv, dict)
        self.t_.set_value(dict['t'])


radam = Radam()


class Adam(Radam):
    def __init__(self):
        super(Adam, self).__init__()

    def init(self, wrt, loss):
        super(Adam, self).init(wrt, loss)

    def get_updates(self):
        self.get_grad()
        cm = OrderedDict()
        cv = OrderedDict()
        self.iter_dict_(lambda x, y: self.beta_1_ * y + (1. - self.beta_1_) * x, self.gparams,
                        self.pm, cm)
        self.iter_dict_(lambda x, y: self.beta_2_ * y + (1. - self.beta_2_) * (x ** 2), self.gparams,
                        self.pv, cv)
        alpha_t = self.learning_rate * (T.sqrt(1. - self.beta_2_ ** self.t_) / (1. - self.beta_1_ ** self.t_))
        self.iter_dict_(lambda x, y: alpha_t * (x / (T.sqrt(y) + self.epsilon_)), cm,
                        cv, self.updates2params)
        self.iter_updates()
        self.iter_register(self.pm, cm, self.updates_m)
        self.iter_register(self.pv, cv, self.updates_v)
        self.updates.update(self.updates_m)
        self.updates.update(self.updates_v)
        self.updates.update({self.t_: self.t_ + 1})
        return self.updates


adam = Adam()


class Nadam(Radam):
    def __init__(self):
        super(Nadam, self).__init__()
        self.beta_1 = 0.99

    def init(self, wrt, loss):
        super(Nadam, self).init(wrt, loss)
        self.a_beta_1_ = kernel.shared(self.numpy_floatX(1), name='a_beta_1_',attr=[None])

    def get_updates(self):
        self.get_grad()
        cbeta_1 = self.beta_1_ * (1. - 0.5 * (0.96 ** (self.t_ / 250.)))
        fbeta_1 = self.beta_1_ * (1. - 0.5 * (0.96 ** ((self.t_ + 1) / 250.)))
        cbeta_1_ = self.a_beta_1_ * cbeta_1
        fbeta_1_ = cbeta_1_ * fbeta_1
        g_e = OrderedDict()
        self.iter_dict(lambda x: x / (1 - cbeta_1_), self.gparams, g_e)
        cm = OrderedDict()
        cv = OrderedDict()
        cm_e = OrderedDict()
        cv_e = OrderedDict()
        cm_bar = OrderedDict()
        self.iter_dict_(lambda x, y: self.beta_1_ * y + (1. - self.beta_1_) * x, self.gparams,
                        self.pm, cm)
        self.iter_dict_(lambda x, y: self.beta_2_ * y + (1. - self.beta_2_) * (x ** 2), self.gparams,
                        self.pv, cv)
        self.iter_dict(lambda x: x / (1. - fbeta_1_), cm, cm_e)
        self.iter_dict(lambda x: x / (1. - self.beta_2_ ** self.t_), cv, cv_e)
        self.iter_dict_(lambda x, y: (1. - cbeta_1) * x + fbeta_1 * y, g_e, cm_e, cm_bar)
        self.iter_dict_(lambda x, y: self.learning_rate * (x / (T.sqrt(y) + self.epsilon_)), cm_bar,
                        cv_e, self.updates2params)
        self.iter_updates()
        self.iter_register(self.pm, cm, self.updates_m)
        self.iter_register(self.pv, cv, self.updates_v)
        self.updates.update(self.updates_m)
        self.updates.update(self.updates_v)
        self.updates.update({self.t_: self.t_ + 1, self.a_beta_1_: self.a_beta_1_ * cbeta_1})
        return self.updates

    def save_(self, dict):
        dict = super(Nadam, self).save_(dict)
        dict['a_beta_1'] = self.a_beta_1_.get()
        return dict

    def load_(self, dict):
        super(Nadam, self).save_(dict)
        self.a_beta_1_.set_value(dict['a_beta_1'])


nadam = Nadam()
