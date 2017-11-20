# -*- coding: utf-8 -*-
"""
Created on  四月 26 21:10 2017

@author: aeloyq
"""

import numpy as np
from utils import *
from param import *
from nnbuilder.kernel import *
from collections import OrderedDict


class Ops(object):
    def __init__(self):
        '''

        '''
        self._ops_list = []

    def append(self, op):
        '''

        :param op:
        :return:
        '''
        self._ops_list.append(op)

    def update(self, op_instance):
        '''

        :param op:
        :return:
        '''
        ops_list = self.get()
        for op in ops_list:
            if isinstance(op_instance, op.op):
                op.update(op_instance.options)
                op.switch = True

    def get(self):
        '''

        :return:
        '''
        return self._ops_list

    def get_on(self):
        '''

        :return:
        '''
        return [op for op in self._ops_list if op.switch]

    def get_off(self):
        '''

        :return:
        '''
        return [op for op in self._ops_list if not op.switch]

    def show(self):
        '''

        :return:
        '''
        return [str(op) for op in self._ops_list]

    def show_on(self):
        '''

        :return:
        '''
        return [str(op) for op in self._ops_list if op.switch]

    def show_off(self):
        '''

        :return:
        '''
        return [str(op) for op in self._ops_list if not op.switch]


class Op(object):
    def __init__(self, op, layer, scan=False, **options):
        '''
        a class of trick operations of Layer
        :param tvar:
        :param kwargs:
        '''
        self.op = op
        self.layer = layer
        self.scan = scan
        self.switch = False
        self.options = options
        self.name = ''

    def __str__(self):
        return self.layer.name + '_' + self.name

    def update(self, options):
        '''

        :param options:
        :return:
        '''
        self.options.update(options)

    def _is_train(self):
        return self.switch and self.layer.root._build_mode == 'train'

    def _is_running(self):
        return self.switch and self.layer.root._build_mode == 'running'

    def apply(self, X, scan_X=None):
        '''
        trick operations of Layer
        usually used to train the layer properly
        this is a pre-register method so that these trick may be applied correctly
        which means tricks will be in recomended order and using right trick paramerter
        Layer may use this operation if it was added by model.add(), otherwise will skip through
        :return: kernel.tensor
            the graph after operation
        '''
        if self._is_train():
            if not self.scan:
                return self.op.op(self.layer, X, self.options)
            else:
                return self.op.scan_op(self.layer, X, scan_X, self.options)
        elif self._is_running():
            if not self.scan:
                return self.op.op_(self.layer, X, self.options)
            else:
                return self.op.scan_op_(self.layer, X, scan_X, self.options)
        else:
            return X

    def scan_share_states(self, sequence, iter_state, non_iter_state, non_sequence):
        if self._is_train():
            share_sequence, share_iter_state, share_non_iter_state, share_non_sequence = self.op.scan_share_states(
                self.layer, sequence,
                iter_state,
                non_iter_state,
                non_sequence,
                self.options)
        else:
            share_sequence, share_iter_state, share_non_iter_state, share_non_sequence = self.op.scan_share_states_(
                self.layer, sequence,
                iter_state,
                non_iter_state,
                non_sequence,
                self.options)
        sequence.update(share_sequence)
        iter_state.update(share_iter_state)
        non_iter_state.extend(share_non_iter_state)
        non_sequence.update(share_non_sequence)

    def scan_add_outputs(self, outputs):
        if self._is_train():
            addition_outputs = self.op.scan_add_outputs(self.options)
        else:
            addition_outputs = self.op.scan_add_outputs_(self.options)
        return outputs + addition_outputs

    def scan_add_updates(self, scan_outputs, scan_updates, n_step, options):
        if self._is_train():
            addition_updates = self.op.scan_add_updates(self.layer, scan_outputs, scan_updates, n_step, options)
        else:
            addition_updates = self.op.scan_add_updates_(self.layer, scan_outputs, scan_updates, n_step, options)
        scan_updates.update(addition_updates)


class Inlayerops(object):
    '''
    operations of layer's inner graph or model's loss function
    such as dropout\weight decay etc.
    '''
    name = 'ops'

    def __init__(self, op_units=None):
        self.name = self.__class__.name
        self.op_units = op_units
        self.options = {}

    @staticmethod
    def op(layer, X, options):
        return X

    @staticmethod
    def op_(layer, X, options):
        return X

    @staticmethod
    def scan_op(layer, X, scan_X, options):
        return X

    @staticmethod
    def scan_op_(layer, X, scan_X, options):
        return X

    @staticmethod
    def scan_share_states(layer, sequence, iter_state, non_iter_state, non_sequence, options):
        return OrderedDict(), OrderedDict(), [], OrderedDict()

    @staticmethod
    def scan_add_outputs(layer, outputs, options):
        return []

    @staticmethod
    def scan_add_updates(layer, scan_outputs, scan_updates, n_step, options):
        return OrderedDict()

    @staticmethod
    def scan_share_states_(layer, sequence, iter_state, non_iter_state, non_sequence, options):
        return OrderedDict(), OrderedDict(), [], OrderedDict()

    @staticmethod
    def scan_add_outputs_(layer, outputs, options):
        return []

    @staticmethod
    def scan_add_updates_(layer, scan_outputs, scan_updates, n_step, options):
        return OrderedDict()


class Betweenlayerops(object):
    '''
    operations of between layers'graph or model's loss function
    such as dropout\weight decay etc.
    '''
    name = 'ops'

    def __init__(self, others=None):
        self.name = self.__class__.name
        self.others = others

    def init(self, layer):
        self.layer = layer

    def op(self):
        pass


class Lossops(object):
    '''
    operations of layer's inner graph or model's loss function
    such as dropout\weight decay etc.
    '''
    name = 'ops'

    def __init__(self):
        pass

    def init(self, layer, model):
        self.layer = layer
        self.model = model

    def build(self, loss):
        return loss


class dropout(Inlayerops):
    name = 'dropout'

    def __init__(self, noise=0.5, op_units=None):
        Inlayerops.__init__(self, op_units)
        self.options = {'noise': noise}

    @staticmethod
    def get_options(X, options):
        if 'broadcast' not in options:
            broadcast = None
        else:
            broadcast = options['broadcast']
        if 'shape' not in options:
            if broadcast is None:
                shape = X.shape
            else:
                shape = X.shape[:-broadcast]
        else:
            shape = options['shape']
        noise = options['noise']
        return broadcast, shape, noise

    @staticmethod
    def apply_mask(X, mask, broadcast):
        if broadcast is None:
            return X * mask
        else:
            return X * T.forwardbroadcastitem(mask, broadcast)

    @staticmethod
    def op(layer, X, options):

        broadcast, shape, noise = dropout.get_options(X, options)
        return dropout.apply_mask(X, R.binomial(shape, p=noise) / noise, broadcast)

    @staticmethod
    def op_(layer, X, options):
        return X

    @staticmethod
    def scan_op(layer, X, scan_X, options):
        if 'broadcast' not in options:
            broadcast = None
        else:
            broadcast = options['broadcast']
        mask = scan_X[options['name']]
        return dropout.apply_mask(X, mask, broadcast)

    @staticmethod
    def scan_op_(layer, X, **options):
        return X

    @staticmethod
    def scan_share_states(layer, sequence, iter_state, non_iter_state, non_sequence, **options):
        addition_non_sequence = OrderedDict()
        name = options['name']
        shape = options['shape']
        noise = options['noise']
        addition_non_sequence[name] = R.binomial(shape, p=noise) / noise
        return OrderedDict(), OrderedDict(), [], addition_non_sequence

    @staticmethod
    def scan_share_states_(layer, sequence, iter_state, non_sequence, **options):
        return OrderedDict(), OrderedDict(), [], OrderedDict()

    @staticmethod
    def scan_add_outputs(layer, outputs, **options):

        return []

    @staticmethod
    def scan_add_outputs_(layer, outputs, **options):
        return []

    @staticmethod
    def scan_add_updates(layer, scan_outputs, scan_updates, n_step, options):
        return OrderedDict()

    @staticmethod
    def scan_add_updates_(layer, scan_outputs, scan_updates, n_step, options):
        return OrderedDict()


class normalization(Inlayerops):
    name = 'normalization'
    method = '[\'layer\',\'batch\']'

    def __init__(self, method='layer', move_average_factor=0.1, epsilon=1e-5, op_units=None):
        Inlayerops.__init__(self, op_units)
        self.options = {'method': method, 'maf': move_average_factor, 'eps': epsilon}

    @staticmethod
    def get_prefix(options):
        return options['name'] + '_' + options['method'] + '_' + normalization.name + '_'

    @staticmethod
    def get_prepare(X, options):
        prefix = normalization.get_prefix(options)

        try:
            batch_axis = X.attr.index('batch')
        except:
            batch_axis = 0
        try:
            unit_axis = X.attr.index('unit')
        except:
            unit_axis = -1

        if 'unit_dim' in options:
            unit_dim = options['unit_dim']
        else:
            unit_dim = [X.size[-1]]

        return prefix, batch_axis, unit_axis, unit_dim

    @staticmethod
    def preallocate_training(X, layer, options):
        prefix, batch_axis, unit_axis, unit_dim = normalization.get_prepare(X, options)

        gamma_name = prefix + 'gamma'
        gamma = Parameter(layer, gamma_name, Parameter.normscale, random=Parameter.ones, shape=unit_dim)
        setattr(layer, gamma_name, gamma)
        beta_name = prefix + 'beta'
        beta = Parameter(layer, beta_name, Parameter.normBias, random=Parameter.zeros, shape=unit_dim)
        setattr(layer, beta_name, beta)
        eps = options['eps']

        return prefix, batch_axis, unit_axis, unit_dim, gamma(), beta(), eps

    @staticmethod
    def preallocate_running(layer, prefix, unit_dim):
        running_prefix = 'running_'
        mu_name = running_prefix + prefix + 'mu'
        sigma_name = running_prefix + prefix + 'sigma'
        if hasattr(layer, mu_name):
            mu = getattr(layer, mu_name)
        else:
            mu = Parameter(layer, mu_name, role=None, random=Parameter.zeros, shape=unit_dim)
            setattr(layer, mu_name, mu)
        if hasattr(layer, sigma_name):
            sigma = getattr(layer, sigma_name)
        else:
            sigma = Parameter(layer, sigma_name, role=None, random=Parameter.ones, shape=unit_dim)
            setattr(layer, sigma_name, sigma)
        return mu(), sigma()

    @staticmethod
    def update_running(layer, X, mu_training, sigma_training, prefix, batch_axis, unit_dim, maf):
        batch_size = X.shape[batch_axis]
        batch_size = T.cast(batch_size, kernel.config.floatX)
        mu, sigma = normalization.preallocate_running(layer, prefix, unit_dim)
        updates = {mu: mu * (1 - maf) + mu_training * maf,
                   sigma: sigma * (1 - maf) + (sigma_training * (batch_size / (batch_size - 1))) * maf}
        layer.updates.update(updates)

    @staticmethod
    def normlize(layer, X, options):
        prefix, batch_axis, unit_axis, unit_dim, gamma, beta, _eps = normalization.preallocate_training(X, layer,
                                                                                                        options)
        if options['method'] == 'batch':
            axis = batch_axis
        else:
            axis = unit_axis
        mu_training = X.mean(axis, keepdims=True)
        sigma_training = X.var(axis, keepdims=True)
        output = (X - mu_training) / T.sqrt(sigma_training + _eps)
        output = gamma * output + beta
        return output, mu_training, sigma_training, prefix, batch_axis, unit_dim

    @staticmethod
    def normlize_(tvar, layer, options):
        prefix, batch_axis, unit_axis, unit_dim, gamma, beta, _eps = normalization.preallocate_training(tvar, layer,
                                                                                                        options)
        if options['method'] == 'batch':
            mu_running, sigma_running = normalization.preallocate_running(layer, prefix, unit_dim)
        else:
            mu_running, sigma_running = tvar.mean(unit_axis, keepdims=True), tvar.var(unit_axis, keepdims=True)
        output = (tvar - mu_running) / T.sqrt(sigma_running + _eps)
        output = gamma * output + beta
        return output

    @staticmethod
    def op(layer, X, options):
        output, mu_training, sigma_training, prefix, batch_axis, unit_dim = normalization.normlize(X, layer, options)
        if options['method'] == 'batch':
            normalization.update_running(layer, X, mu_training, sigma_training, prefix, batch_axis, unit_dim,
                                         options['ma'])
        return output

    @staticmethod
    def op_(layer, X, **kwargs):
        output = normalization.normlize_(X, layer, **kwargs)
        return output

    @staticmethod
    def scan_op(layer, X, scan_X, options):
        output, mu_training, sigma_training, prefix, batch_axis, unit_dim = normalization.normlize(X, layer, **options)
        options['mu_training'] = mu_training
        options['sigma_training'] = sigma_training
        options['prefix'] = prefix
        options['unit_dim'] = unit_dim
        options['X'] = X
        return output

    @staticmethod
    def scan_op_(layer, X, scan_X, options):
        output = normalization.normlize_(layer, X, options)
        return output

    @staticmethod
    def scan_share_states(layer, sequence, iter_state, non_iter_state, non_sequence, options):
        addition_non_iter_state = [options['name'] + 'mu_training', options['name'] + 'sigma_training']
        return OrderedDict(), OrderedDict(), addition_non_iter_state, OrderedDict()

    @staticmethod
    def scan_add_outputs(layer, outputs, options):
        addition_outputs = [options['mu_training'], options['sigma_training']]
        return addition_outputs

    @staticmethod
    def scan_add_updates(layer, scan_outputs, scan_updates, n_step, options):
        mu_training = scan_outputs[options['name'] + 'mu_training'] / T.cast(n_step, kernel.config.floatX)
        sigma_training = scan_outputs[options['name'] + 'sigma_training'] / T.cast(n_step, kernel.config.floatX)
        if options['method'] == 'batch':
            normalization.update_running(layer, options['X'], mu_training, sigma_training, options['prefix'],
                                         options['batch_axis'], options['unit_dim'], options['ma'])

    @staticmethod
    def scan_add_outputs_(layer, outputs, **options):
        return []

    @staticmethod
    def scan_add_updates(layer, scan_outputs, scan_updates, n_step, options):
        return OrderedDict()

    @staticmethod
    def scan_add_updates_(layer, scan_outputs, scan_updates, n_step, options):
        return OrderedDict()


class lrn(Inlayerops):
    name = 'lrn'

    def __init__(self, k=2, n=5, alpha=1e-4, beta=0.75, op_units=None):
        Inlayerops.__init__(self, op_units)
        self.dict = {'k': k, 'n': n, 'alpha': alpha, 'beta': beta}

    @staticmethod
    def op(layer, X, options):
        half = options['n'] // 2
        sq = T.sqr(X)
        if X.ndim == 4:
            b, ch, r, c = X.shape
            tvar_padding = T.alloc.alloc(0., [b, ch + 2 * half, r, c], X.attr)
            tvar_padding[:, half:half + ch, :, :] = sq
            norm_factor = options['k']
            for i in range(options['n']):
                norm_factor += options['alpha'] * sq[:, i:i + ch, :, :]
            norm_factor = norm_factor ** options['beta']
        else:
            b, ch, r, c, d = X.shape
            tvar_padding = T.alloc.alloc(0., [b, ch + 2 * half, r, c, d], X.attr)
            tvar_padding[:, half:half + ch, :, :, :] = sq
            norm_factor = options['k']
            for i in range(options['n']):
                norm_factor += options['alpha'] * sq[:, i:i + ch, :, :, :]
            norm_factor = norm_factor ** options['beta']
        return X / norm_factor

    @staticmethod
    def op_(layer, X, options):
        lrn.op(layer, X, options)


class weightnorm(Inlayerops):
    name = 'weightnorm'

    def __init__(self, epsilon=1e-5, op_units=None):
        Inlayerops.__init__(self, op_units)
        self.options['eps'] = epsilon

    @staticmethod
    def normalize(layer, X, options):
        prefix = options['name'] + '_' + weightnorm.name + '_'
        gamma_name = prefix + 'gamma'
        eps = options['epsilon']
        if 'unit_dim' in options:
            unit_dim = options['unit_dim']
        else:
            unit_dim = X.size[-1]
        if 'slices' in options:
            slices = options['slices']
            raw_shape = X.shape
            new_shape = []
            for i in range(X.ndim - 1):
                new_shape.append(X.shape[i])
            new_shape.extend([slices, X.shape[-1] // slices])
            unit_dim = unit_dim // slices
            X = X.reshape(new_shape)
            gamma = Parameter(layer, gamma_name, Parameter.normscale, random=Parameter.ones, shape=unit_dim)
            setattr(layer, gamma_name, gamma)
            X = X / T.sqrt(X.var(-1, keepdims=True) + eps)
            X = X * gamma
            X = X.reshape(raw_shape)
        else:
            gamma = Parameter(layer, gamma_name, Parameter.normscale, random=Parameter.ones, shape=unit_dim)
            setattr(layer, gamma_name, gamma)
            X = X / T.sqrt((X.var(-1, keepdims=True) + eps))
            X = X * gamma
        return X

    @staticmethod
    def op(layer, X, options):
        return weightnorm.normalize(layer, X, options)

    @staticmethod
    def op_(layer, X, options):
        return weightnorm.normalize(layer, X, options)

    @staticmethod
    def scan_op(layer, X, scan_X, options):
        return weightnorm.normalize(layer, X, options)

    @staticmethod
    def scan_op_(layer, X, scan_X, options):
        return weightnorm.normalize(layer, X, options)

    @staticmethod
    def scan_share_states(layer, sequence, iter_state, non_iter_state, non_sequence, options):
        return OrderedDict(), OrderedDict(), [], OrderedDict()

    @staticmethod
    def scan_add_outputs(layer, outputs, options):
        return []

    @staticmethod
    def scan_add_updates(layer, scan_outputs, scan_updates, n_step, options):
        return OrderedDict()

    @staticmethod
    def scan_share_states_(layer, sequence, iter_state, non_iter_state, non_sequence, options):
        return OrderedDict(), OrderedDict(), [], OrderedDict()

    @staticmethod
    def scan_add_outputs_(layer, outputs, options):
        return []

    @staticmethod
    def scan_add_updates_(layer, scan_outputs, scan_updates, n_step, options):
        return OrderedDict()


class weight_decay(Lossops):
    name = 'weight_decay'

    def __init__(self, alpha=0.0001, params=None):
        Lossops.__init__(self)
        self.alpha = alpha
        self.params = params

    def build(self, loss):
        reg = 0
        if self.params is None:
            params = self.layer.params.values()
        else:
            params = self.params
        for param in params:
            if param.role is Parameter.weight:
                reg += param().sqr().sum()
        return loss + self.alpha * reg


class regularization(Lossops):
    name = 'regularization'

    def __init__(self, alpha=0.0001):
        Lossops.__init__(self)
        self.alpha = alpha

    def build(self, loss):
        reg = 0
        for layer in self.model.layers.values():
            for param in layer.params.values():
                if param.role is Parameter.weight:
                    reg += param().sqr().sum()
        return loss + self.alpha * reg


class Residual(Betweenlayerops):
    name = 'residual'

    def __init__(self, others):
        Betweenlayerops.__init__(self, others)

    def op(self):
        self.layer.output = self.layer.output + self.others.output
        self.layer.outputs['output'] = self.layer.output
        self.layer.running_output = self.layer.running_output + self.others.running_output
        self.layer.running_outputs['output'] = self.layer.running_output
