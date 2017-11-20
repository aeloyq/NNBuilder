# -*- coding: utf-8 -*-
"""
Created on Sat Dec 17 13:55:42 2016

@author: aeloyq
"""
from basic import *


class Linear(LayerBase):
    def __init__(self, unit, **kwargs):
        '''

        :param unit:
        :param kwargs:
        '''
        LayerBase.__init__(self, **kwargs)
        self.unit_dim = unit

    def set_params(self):
        self.weight = Parameter(self, 'Weight', Parameter.weight, random=Parameter.uniform,
                                shape=[self.in_dim, self.unit_dim])

    def apply(self, X):
        return T.dot(X, self.weight())


class Scale(LayerBase):
    def __init__(self, unit=None, **kwargs):
        '''

        :param unit:
        :param kwargs:
        '''
        LayerBase.__init__(self, **kwargs)
        self.unit_dim = unit

    def set_params(self):
        self.scaleweight = Parameter(self, 'ScaleWeight', Parameter.scalew, random=Parameter.uniform)
        if self.unit_dim is None:
            self.unit_dim = self.in_dim
        if isinstance(self.unit_dim, (list, tuple)):
            self.scaleweight.shape = self.unit_dim
        else:
            self.scaleweight.shape = [self.unit_dim]

    def apply(self, X):
        return X * self.scaleweight()


class Sparse(LayerBase):
    pass


class Bias(LayerBase):
    def __init__(self, unit=None, **kwargs):
        '''

        :param kwargs:
        '''
        LayerBase.__init__(self, **kwargs)
        self.unit_dim = unit

    def set_params(self):
        self.bias = Parameter(self, 'Bias', Parameter.bias, random=Parameter.uniform)
        if self.unit_dim is None:
            self.unit_dim = self.in_dim
        self.bias.shape = [self.unit_dim]

    def apply(self, X):
        return X + self.bias()


class Activation(LayerBase):
    def __init__(self, activation=T.tanh, **kwargs):
        '''

        :param activation:
        :param kwargs:
        '''
        LayerBase.__init__(self, **kwargs)
        self.activation = activation

    def apply(self, X):
        return self.activation(X)


class LinearBias(LayerBase):
    def __init__(self, unit, **kwargs):
        '''

        :param unit:
        :param kwargs:
        '''
        LayerBase.__init__(**kwargs)
        self.unit = unit

    def set_children(self):
        self.linear = Linear(self.unit)
        self.bias = Bias(self.unit)

    def apply(self, X):
        linear_output = self.linear.apply(X)
        bias_linear_output = self.bias.apply(linear_output)
        return bias_linear_output


class LinearActivation(LayerBase):
    def __init__(self, unit, activation=T.tanh, **kwargs):
        '''

        :param unit:
        :param activation:
        :param kwargs:
        '''
        LayerBase.__init__(**kwargs)
        self.unit = unit
        self.activation = activation

    def set_children(self):
        self.linear = Linear(self.unit)
        self.activation = Activation(self.activation)

    def apply(self, X):
        linear_output = self.linear.apply(X)
        activation_linear_output = self.activation.apply(linear_output)
        return activation_linear_output


class LinearBiasActivation(LayerBase):
    def __init__(self, unit, activation=T.tanh, **kwargs):
        '''

        :param unit:
        :param activation:
        :param kwargs:
        '''
        LayerBase.__init__(self, **kwargs)
        self.unit = unit
        self.activated = activation

    def set_children(self):
        self.linear = Linear(self.unit)
        self.bias = Bias(self.unit)
        self.activation = Activation(self.activated)

    def set_params(self):
        self.bias.shape = [self.linear.unit_dim]

    def apply(self, X):
        linear_output = self.linear.apply(X)
        bias_linear_output = self.bias.apply(linear_output)
        activation_bias_linear_output = self.activation.apply(bias_linear_output)
        return activation_bias_linear_output


class Lookup(LayerBase):
    def __init__(self, unit, vocab, **kwargs):
        '''

        :param unit:
        :param kwargs:
        '''
        LayerBase.__init__(self, **kwargs)
        self.emb_dim = unit
        self.vocab_dim = vocab

    def set_params(self):
        self.table = Parameter(self, 'Table', Parameter.Table, random=Parameter.randn)
        self.table.shape = [self.vocab_dim, self.emb_dim]

    def apply(self, X):
        return T.lookup(X, self.table())


class Hidden(LayerBase):
    def __init__(self, unit, bias=True, activation=T.tanh, **kwargs):
        '''

        :param unit:
        :param bias:
        :param activation:
        :param kwargs:
        '''
        LayerBase.__init__(self, **kwargs)
        self.biased = bias
        self.activated = activation
        self.unit = unit

    def set_children(self):
        self.linear = Linear(self.unit)
        if self.biased:
            self.bias = Bias(self.unit)

        if self.activated is not None:
            self.activation = Activation(activation=self.activated)

    def apply(self, X):
        output = self.linear.apply(X)
        if self.biased:
            output = self.bias.apply(output)
        if self.activated:
            output = self.activation.apply(output)
        return output


class Dense(LayerBase):
    def __init__(self, unit, bias=True, **kwargs):
        '''

        :param unit:
        :param bias:
        :param kwargs:
        '''
        LayerBase.__init__(**kwargs)
        self.biased = bias
        self.unit = unit

    def set_children(self):
        self.linear = Linear(self.unit)
        if self.biased:
            self.bias = Bias(self.unit)

    def apply(self, X):
        output = self.linear.apply(X)
        if hasattr(self, 'bias'):
            output = self.bias.apply(output)
        return output


class Embedding(LayerBase):
    def __init__(self, unit, vocab, **kwargs):
        '''

        :param unit:
        :param kwargs:
        '''
        LayerBase.__init__(self, **kwargs)
        self.unit = unit
        self.vocab = vocab

    def set_children(self):
        self.lookup = Lookup(self.unit, self.vocab)

    def apply(self, X):
        return self.lookup.apply(X)


class MultiEmbedding(LayerBase):
    def __init__(self, names, units, vocabs, **kwargs):
        '''

        :param units:
        :param kwargs:
        '''
        LayerBase.__init__(self, **kwargs)
        self.names = names
        self.units = units
        self.vocabs = vocabs

    def set_children(self):
        lookups = OrderedDict()
        for name, unit, vocab in self.names, self.units, self.vocabs:
            lookups[name] = Lookup(unit, vocab)
        self.lookups = LayerDict(self, lookups)

    def apply(self, X):
        embedding_list = []
        for lookup in self.lookups:
            embedding_list.append(lookup.apply(X))
        return T.concatenate(embedding_list, axis=X.ndim)


class Logistic(LinearBiasActivation):
    def __init__(self, unit, **kwargs):
        LinearBiasActivation.__init__(self, unit, T.sigmoid, **kwargs)

    def set_params(self):
        LinearBiasActivation.set_params(self)
        self.linear.weight.random = Parameter.scaled_uniform


class Softmax(LinearBiasActivation):
    def __init__(self, unit, **kwargs):
        LinearBiasActivation.__init__(self, unit, T.softmax, **kwargs)

    def set_params(self):
        LinearBiasActivation.set_params(self)
        self.linear.weight.random = Parameter.randn

    def sample(self, output):
        return {'predict': T.argmax(output, -1)}


class Dropout(LayerBase):
    def __init__(self, noise=0.5, **kwargs):
        LayerBase.__init__(self, **kwargs)
        self.noise = noise

    def set_ops(self):
        self.dropout = Op(self, dropout, {'noise': self.noise})
        self.dropout.switch = True

    def apply(self, X):
        return self.dropout.apply(X)


class DropoutEmbedding(LayerBase):
    def __init__(self, noise=0.5, **kwargs):
        LayerBase.__init__(self, **kwargs)
        self.noise = noise

    def set_ops(self):
        self.dropout = Op(self, dropout, {'noise': self.noise, 'broadcast': 1})
        self.dropout.switch = True

    def apply(self, X):
        return self.dropout.apply(X)


class DropoutChannel2D(LayerBase):
    def __init__(self, noise=0.5, **kwargs):
        LayerBase.__init__(self, **kwargs)
        self.noise = noise

    def set_ops(self):
        self.dropout = Op(self, dropout, {'noise': self.noise, 'broadcast': 2})
        self.dropout.switch = True

    def apply(self, X):
        return self.dropout.apply(X)


class DropoutChannel3D(LayerBase):
    def __init__(self, noise=0.5, **kwargs):
        LayerBase.__init__(self, **kwargs)
        self.noise = noise

    def set_ops(self):
        self.dropout = Op(self, dropout, {'noise': self.noise, 'broadcast': 3})
        self.dropout.switch = True

    def apply(self, X):
        return self.dropout.apply(X)


class BatchNorm(LayerBase):
    def __init__(self, move_average_factor=0.1, epsilon=1e-5, **kwargs):
        LayerBase.__init__(**kwargs)
        self.maf=move_average_factor
        self.eps=epsilon

    def set_ops(self):
        self.batchnorm = Op(self, normalization, {'method': 'batch', 'maf': self.maf, 'eps': self.eps})
        self.batchnorm.switch = True


    def apply(self, X):
        return self.batchnorm.apply(X)


class BatchNorm2D(LayerBase):
    def __init__(self, move_average_factor=0.1, epsilon=1e-5, **kwargs):
        LayerBase.__init__(**kwargs)
        self.maf=move_average_factor
        self.eps=epsilon

    def set_ops(self):
        self.batchnorm = Op(self, normalization, {'method': 'batch', 'maf': self.maf, 'eps': self.eps})
        self.batchnorm.switch = True

    def apply(self, X):
        self.batchnorm.update({'unit_dim': X.size[-3:]})
        return self.batchnorm.apply(X)


class BatchNorm3D(LayerBase):
    def __init__(self, move_average_factor=0.1, epsilon=1e-5, **kwargs):
        LayerBase.__init__(**kwargs)
        self.maf=move_average_factor
        self.eps=epsilon

    def set_ops(self):
        self.batchnorm = Op(self, normalization, {'method': 'batch', 'maf': self.maf, 'eps': self.eps})
        self.batchnorm.switch = True

    def apply(self, X):
        self.batchnorm.update({'unit_dim': X.size[-4:]})
        return self.batchnorm.apply(X)


class LayerNorm(LayerBase):
    def __init__(self, epsilon=1e-5, **kwargs):
        LayerBase.__init__(**kwargs)
        self.eps=epsilon

    def set_ops(self):
        self.batchnorm = Op(self, normalization, {'method': 'layer', 'eps': self.eps})
        self.batchnorm.switch = True

    def apply(self, X):
        return self.batchnorm.apply(X)
