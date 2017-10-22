# -*- coding: utf-8 -*-
"""
Created on  四月 26 21:10 2017

@author: aeloyq
"""

import numpy as np
from utils import *
from roles import *
from collections import OrderedDict
from nnbuilder.kernel import *


class inlayerops(object):
    '''
    operations of layer's inner graph or model's loss function
    such as dropout\weight decay etc.
    '''
    name = 'ops'

    def __init__(self, units=None):
        self.name = self.__class__.name
        self.units = units
        self.dict = {}

    def init(self, layer):
        self.layer = layer

    def build(self, layer, cur_layer_ops_option):
        self.layer = layer
        if self.units is None:
            if self.name not in layer.ops:
                layer.ops.append(self.name)
            cur_layer_ops_option['default'] = self.dict
        else:
            for unit in self.units:
                if unit + '_' + self.name not in layer.ops:
                    layer.ops.append(unit + '_' + self.name)
                cur_layer_ops_option[unit] = self.dict

    @staticmethod
    def op(layer, tvar, **kwargs):
        pass

    @staticmethod
    def op_(layer, tvar, **kwargs):
        pass


class betweenlayerops(object):
    '''
    operations of between layers'graph or model's loss function
    such as dropout\weight decay etc.
    '''
    name = 'ops'

    def __init__(self, units=None):
        self.name = self.__class__.name
        self.units = units
        self.dict = {}

    def init(self, layer):
        self.layer = layer

    def build(self, layer, cur_layer_ops_option):
        self.layer = layer
        if self.units is None:
            if self.name not in layer.ops:
                layer.ops.append(self.name)
            cur_layer_ops_option['default'] = self.dict
        else:
            for unit in self.units:
                if unit + '_' + self.name not in layer.ops:
                    layer.ops.append(unit + '_' + self.name)
                cur_layer_ops_option[unit] = self.dict

    @staticmethod
    def op(layer, tvar, **kwargs):
        pass

    @staticmethod
    def op_(layer, tvar, **kwargs):
        pass


class lossops(object):
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

    def build(self, loss, **kwargs):
        return loss


class dropout(inlayerops):
    name = 'dropout'

    def __init__(self, noise=0.5, units=None):
        inlayerops.__init__(self, units)
        self.dict = {'noise': noise}

    @staticmethod
    def op(layer, tvar, **kwargs):

        def apply_mask(tvar, mask, broadcast):
            if broadcast is None:
                return tvar * mask
            else:
                return tvar * T.forwardbroadcastitem(mask, -broadcast)

        if 'step' in kwargs:
            if kwargs['step'] == 'share':
                shape = layer.initiate_states.values()[0].shape
                for name in layer.inscan_param:
                    tvar[name + 'Mask'] = R.binomial(shape, p=kwargs['noise']) / kwargs['noise']
                return tvar
            elif kwargs['step'] == 'apply':
                return tvar * kwargs['inputs'][kwargs['pname'] + 'Mask']
        else:
            if 'shape' not in kwargs:
                shape = tvar.shape
            else:
                shape = kwargs['shape']
            if 'broadcast' not in kwargs:
                broadcast = None
            else:
                broadcast = kwargs['broadcast']
            noise = kwargs['noise']
            if not isinstance(tvar, list):
                return apply_mask(tvar, R.binomial(shape, p=noise) / noise,
                                  broadcast)

    @staticmethod
    def op_(layer, tvar, **kwargs):
        if 'step' in kwargs:
            if kwargs['step'] == 'share':
                return {}
            else:
                return tvar
        else:
            return tvar


class normalization(inlayerops):
    name = 'normalization'
    normdict = '[\'layer\',\'batch\']'

    def __init__(self, method='layer', units=None):
        inlayerops.__init__(self, units)
        self.dict = {'method': method}

    @staticmethod
    def get_name(tvar, layer, **kwargs):
        prefix = kwargs['oname'] + '_' + kwargs['method'] + '_' + normalization.name + '_'

        batch_axis = tvar.attr.index(batch)

        unit_axis = tvar.attr.index(unit)

        if 'unit_dim' in kwargs:
            unit_dim = kwargs['unit_dim']
        else:
            unit_dim = layer.units_dim[kwargs['oname']]

        return prefix, batch_axis, unit_axis, unit_dim

    @staticmethod
    def pop(tvar, layer, **kwargs):
        prefix, batch_axis, unit_axis, unit_dim = normalization.get_name(tvar, layer, **kwargs)

        gamma_name = prefix + 'gamma'
        gamma = layer.allocate(param_init_functions.ones, gamma_name, normweight, unit_dim)
        if kwargs['method'] == 'layer':
            beta_name = prefix + 'beta'
            full_beta_name = layer.name + '_' + beta_name
            if full_beta_name in layer.trainable_params:
                beta = layer.trainable_params[full_beta_name]
            else:
                beta = layer.allocate(param_init_functions.zeros, beta_name, bias, unit_dim)
        else:
            beta = None
        _eps = 1e-5

        return prefix, batch_axis, unit_axis, unit_dim, gamma, beta, _eps

    @staticmethod
    def op(layer, tvar, **kwargs):

        def set_oparam(mu_, sigma_, prefix, batch_axis, unit_dim):
            layer_prefix = layer.name + '_'
            m = tvar.shape[batch_axis]
            fm = T.cast(m, kernel.config.floatX)
            if layer_prefix + prefix + 'n' not in layer.untrainable_params:
                n = kernel.shared(value=0,
                                  name=layer_prefix + prefix + 'n', attr=[None])
                layer.untrainable_params[layer_prefix + prefix + 'n'] = n
            else:
                n = layer.untrainable_params[layer_prefix + prefix + 'n']
            if layer_prefix + prefix + 'mu' not in layer.untrainable_params:
                mu = kernel.shared(value=np.zeros([unit_dim], dtype=kernel.config.floatX),
                                   name=layer_prefix + prefix + 'mu', attr=[unit])
                layer.untrainable_params[layer_prefix + prefix + 'mu'] = mu
            else:
                mu = layer.untrainable_params[layer_prefix + prefix + 'mu']
            if layer_prefix + prefix + 'sigma' not in layer.untrainable_params:
                sigma = kernel.shared(value=np.zeros([unit_dim], dtype=kernel.config.floatX),
                                      name=layer_prefix + prefix + 'sigma', attr=[unit])
                layer.untrainable_params[layer_prefix + prefix + 'sigma'] = sigma
            else:
                sigma = layer.untrainable_params[layer_prefix + prefix + 'sigma']
            updates = {mu.graph: (mu + mu_).graph, sigma.graph: (sigma + (sigma_ * fm) / (fm - 1)).graph,
                       n.graph: (n + 1).graph}
            layer.updates.update(updates)

        def batch_norm(tvar):
            prefix, batch_axis, unit_axis, unit_dim, gamma, beta, _eps = normalization.pop(tvar, layer, **kwargs)
            mu_ = tvar.mean(range(0, batch_axis + 1))
            sigma_ = tvar.var(range(0, batch_axis + 1))
            output = (tvar - mu_) / T.sqrt(
                (sigma_ + _eps))
            return gamma * output, mu_, sigma_, prefix, batch_axis, unit_dim

        def layer_norm(tvar):
            prefix, batch_axis, unit_axis, unit_dim, gamma, beta, _eps = normalization.pop(tvar, layer, **kwargs)
            tvar = tvar + beta
            output = (tvar - tvar.mean(unit_axis, keepdims=True)) / T.sqrt(
                (tvar.var(unit_axis, keepdims=True) + _eps))
            return gamma * output

        if 'step' in kwargs:
            step = kwargs['step']
            if step == 'share':
                if kwargs['method'] == 'batch':
                    output = []
                    for name in layer.inscan_param:
                        output.extend([name + 'Mean', name + 'Std'])
                else:
                    output = []
            elif step == 'apply':
                if kwargs['method'] == 'batch':
                    output, mu_, sigma_, prefix, batch_axis, unit_dim = batch_norm(tvar)
                    pname = kwargs['pname']
                    layer.shared_info['step_batch_norm'] = {pname + 'Mean': mu_, pname + 'Std': sigma_}
                    layer.shared_info['step_batch_info'] = OrderedDict()
                    layer.shared_info['step_batch_info'][pname] = {'unit_name': kwargs['oname'],
                                                                   'unit_dim': layer.units_dim[kwargs['oname']]}
                elif kwargs['method'] == 'layer':
                    output = layer_norm(tvar)
                else:
                    raise NameError
            elif step == 'add':
                if kwargs['method'] == 'batch':
                    output = tvar + layer.shared_info['step_batch_norm'].values()
                else:
                    output = tvar
            elif step == 'update':
                if kwargs['method'] == 'batch':
                    output = None
                    outputs = kwargs['outputs']
                    for name in layer.inscan_param:
                        mu_, sigma_, = outputs[name + 'Mean'], outputs[name + 'Std']
                        mu_ = mu_.mean(mu_.attr.index(time))
                        sigma_ = sigma_.mean(sigma_.attr.index(time))
                        prefix = layer.shared_info['step_batch_info'][name]['unit_name'] + '_' + kwargs[
                            'method'] + '_' + normalization.name + '_'
                        unit_dim = layer.shared_info['step_batch_info'][name]['unit_dim']
                        layer_prefix = layer.name + '_'
                        m = layer.batch_size
                        fm = T.cast(m, kernel.config.floatX)
                        if layer_prefix + prefix + 'n' not in layer.untrainable_params:
                            n = kernel.shared(value=0,
                                              name=layer_prefix + prefix + 'n', attr=[None])
                            layer.untrainable_params[layer_prefix + prefix + 'n'] = n
                        else:
                            n = layer.untrainable_params[layer_prefix + prefix + 'n']
                        if layer_prefix + prefix + 'mu' not in layer.untrainable_params:
                            mu = kernel.shared(value=np.zeros([unit_dim], dtype=kernel.config.floatX),
                                               name=layer_prefix + prefix + 'mu', attr=[unit])
                            layer.untrainable_params[layer_prefix + prefix + 'mu'] = mu
                        else:
                            mu = layer.untrainable_params[layer_prefix + prefix + 'mu']
                        if layer_prefix + prefix + 'sigma' not in layer.untrainable_params:
                            sigma = kernel.shared(value=np.zeros([unit_dim], dtype=kernel.config.floatX),
                                                  name=layer_prefix + prefix + 'sigma', attr=[unit])
                            layer.untrainable_params[layer_prefix + prefix + 'sigma'] = sigma
                        else:
                            sigma = layer.untrainable_params[layer_prefix + prefix + 'sigma']
                        updates = {mu.graph: (mu + mu_).graph,
                                   sigma.graph: (sigma + (sigma_ * fm) / (fm - 1)).graph,
                                   n.graph: (n + 1).graph}
                        layer.updates.update(updates)
                else:
                    output = None
            else:
                raise NameError
        else:
            if kwargs['method'] == 'batch':
                output, mu_, sigma_, prefix, batch_axis, unit_dim = batch_norm(tvar)
                set_oparam(mu_, sigma_, prefix, batch_axis, unit_dim)
            elif kwargs['method'] == 'layer':
                output = layer_norm(tvar)
            else:
                raise NameError

        return output

    @staticmethod
    def op_(layer, tvar, **kwargs):

        def batch_norm(tvar):
            prefix, batch_axis, unit_axis, unit_dim, gamma, beta, _eps = normalization.pop(tvar, layer, **kwargs)
            layer_prefix = layer.name + '_'
            n = layer.untrainable_params[layer_prefix + prefix + 'n']
            mu_ = layer.untrainable_params[layer_prefix + prefix + 'mu']
            sigma_ = layer.untrainable_params[layer_prefix + prefix + 'sigma']
            n = T.switch(T.eq(n, 0), 1, n)
            mu = mu_ / T.cast(n, kernel.config.floatX)
            sigma = sigma_ / T.cast(n, kernel.config.floatX)
            output = (tvar - mu) / T.sqrt((sigma + _eps))
            output = gamma * output
            return output

        def layer_norm(tvar):
            prefix, batch_axis, unit_axis, unit_dim, gamma, beta, _eps = normalization.pop(tvar, layer, **kwargs)
            tvar = tvar + beta
            output = (tvar - tvar.mean(unit_axis, keepdims=True)) / T.sqrt(
                (tvar.var(unit_axis, keepdims=True) + _eps))
            output = gamma * output
            return output

        if 'step' not in kwargs:
            if kwargs['method'] == 'batch':
                output = batch_norm(tvar)
            elif kwargs['method'] == 'layer':
                output = layer_norm(tvar)
            else:
                raise NameError
        else:
            step = kwargs['step']
            if step == 'share':
                output = []
            elif step == 'apply':
                if kwargs['method'] == 'batch':
                    output = batch_norm(tvar)
                elif kwargs['method'] == 'layer':
                    output = layer_norm(tvar)
                else:
                    raise NameError
            elif step == 'add':
                output = tvar
            elif step == 'update':
                output = None
            else:
                output = None
        return output


class weightnorm(inlayerops):
    name = 'weightnorm'

    def __init__(self, units=None):
        inlayerops.__init__(self, units)

    @staticmethod
    def pop(layer, tvar, **kwargs):
        prefix = kwargs['oname'] + '_' + weightnorm.name + '_'
        gamma_name = prefix + 'gamma'
        _eps = 1e-5
        if 'slices' in kwargs:
            slices = kwargs['slices']
            raw_shape = tvar.get().shape
            new_shape = []
            for i in range(tvar.ndim - 1):
                new_shape.append(tvar.shape[i])
            new_shape.extend([slices, tvar.shape[-1] // slices])
            unit_dim = raw_shape[-1] // slices
            tvar = tvar.reshape(new_shape)
            gamma = layer.allocate(param_init_functions.ones, gamma_name, normweight, slices, unit_dim)
            tvar = tvar / T.sqrt((tvar.var(-1, keepdims=True) + _eps))
            tvar = tvar * gamma
            tvar = tvar.reshape(raw_shape)
        else:
            gamma = layer.allocate(param_init_functions.ones, gamma_name, weight, tvar.get().shape[-1])
            tvar = tvar / T.sqrt((tvar.var(-1, keepdims=True) + _eps))
            tvar = tvar * gamma
        return tvar

    @staticmethod
    def op(layer, tvar, **kwargs):
        return weightnorm.pop(layer, tvar, **kwargs)

    @staticmethod
    def op_(layer, tvar, **kwargs):
        return weightnorm.pop(layer, tvar, **kwargs)


class weight_decay(lossops):
    name = 'weight_decay'

    def __init__(self, noise=0.0001, params=None):
        lossops.__init__(self)
        self.l2 = noise
        self.params = params

    def build(self, loss, **kwargs):
        reg = 0
        params = []
        pdict = OrderedDict()
        if self.params is None:
            pdict = self.layer.trainable_params
        else:
            for pname in params:
                for name, param in self.layer.trainable_params.items():
                    if pname == name:
                        pdict[name] = param

        for name, param in pdict.items():
            if self.layer.trainable_roles[name] == weight:
                params.append(param)
        for param in params:
            reg += (param ** 2).sum()
        return loss + self.l2 * reg


class regularization(lossops):
    name = 'regularization'

    def __init__(self, noise=0.0001):
        lossops.__init__(self)
        self.l2 = noise

    def build(self, loss, **kwargs):
        reg = 0.
        params = []
        for lname, node in self.model.layers.items():
            for name, param in node.trainable_params.items():
                if node.trainable_roles[name] == weight:
                    params.append(param)
        for param in params:
            reg += (param ** 2).sum()
        return loss + self.l2 * reg


'''
class residual(inlayerops):
    name = 'residual'

    def __init__(self, prelayer=None):
        ops.__init__(self)
        self.prelayer = prelayer

    def build(self, layer, ops_option):
        opsname = residual.name
        if opsname not in ops_option:
            ops_option[opsname]=OrderedDict()
        self.layer = layer
        if self.units is None:
            ops_option[opsname]['default'] = {'prelayer': self.prelayer}
            layer.ops.append(opsname)
        else:
            for unit in self.units:
                ops_option[opsname][unit] = {'prelayer': self.prelayer}
                layer.ops.append(unit + '_' + opsname)

    @staticmethod
    def op(layer, tvar, **kwargs):
        return tvar + kwargs['pre_tvar']

    @staticmethod
    def op_(layer, tvar, **kwargs):
        return residual.op(tvar, **kwargs)
'''
