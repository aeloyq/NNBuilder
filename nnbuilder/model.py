# -*- coding: utf-8 -*-
"""
Created on  Feb 13 10:24 PM 2017

@author: aeloyq
"""
import copy
import extensions
import numpy as np
from data import *
from optimizers import *
from layers.basic import *
from logger import logger
from main import MainLoop, config
from collections import OrderedDict


class X:
    class default:
        def __init__(self, dim):
            self.dim = dim
            self.X = kernel.placeholder('X', [None, dim], ['batch', 'feature'], kernel.config.floatX)

    class num:
        def __init__(self, dim):
            self.dim = dim
            self.X = kernel.placeholder('X', [None, dim], ['batch', 'feature'], kernel.config.floatX)

    class seq:
        def __init__(self, dim=None):
            self.dim = dim
            self.X = kernel.placeholder('X', [dim, None], ['time', 'batch'], kernel.config.floatX)

    class text:
        def __init__(self, dim=None):
            self.dim = dim
            self.X = kernel.placeholder('X', [dim, None], ['time', 'batch'], kernel.config.catX)

    class img:
        def __init__(self, dim):
            self.dim = dim
            self.X = kernel.placeholder('X', [None] + dim, ['batch', 'channel', 'height', 'width'],
                                        kernel.config.floatX)

    class img2d:
        def __init__(self, dim):
            self.dim = dim
            self.X = kernel.placeholder('X', [None] + dim, ['batch', 'channel', 'height', 'width'],
                                        kernel.config.floatX)

    class img3d:
        def __init__(self, dim):
            self.dim = dim
            self.X = kernel.placeholder('X', [None] + dim, ['batch', 'channel', 'deepth', 'height', 'width'],
                                        kernel.config.floatX)


class Y:
    class default:
        def __init__(self):
            self.dim = None
            self.Y = kernel.placeholder('Y', [None], ['batch'], kernel.config.catX)

    class num:
        def __init__(self):
            self.dim = None
            self.Y = kernel.placeholder('Y', [None], ['batch'], kernel.config.floatX)

    class cat:
        def __init__(self):
            self.dim = None
            self.Y = kernel.placeholder('Y', [None], ['batch'], kernel.config.catX)

    class seq:
        def __init__(self, dim=None):
            self.dim = dim
            self.Y = kernel.placeholder('Y', [dim, None], ['time', 'batch'], kernel.config.floatX)

    class text:
        def __init__(self, dim=None):
            self.dim = dim
            self.Y = kernel.placeholder('Y', [dim, None], ['time', 'batch'], kernel.config.catX)

    class mulcat:
        def __init__(self, dim):
            self.dim = dim
            self.Y = kernel.placeholder('Y', [None, dim], ['batch', 'unit'], kernel.config.catX)


class Input(LayerBase):
    def __init__(self, input):
        LayerBase.__init__(self)
        self.name = 'input'
        self.input = input.X
        self.output = self.input
        self.running_output = self.input
        self.in_dim = input.dim
        self.out_dim = self.in_dim


class lossfunctions:
    @staticmethod
    def commen(loss, callback):
        loss = callback(loss)
        if loss.ndim > 0:
            return T.mean(loss)
        else:
            return loss

    @staticmethod
    def binary_crossentropy(outputs, y_true):
        loss = T.binary_crossentropy(outputs['output'], y_true)
        return loss

    @staticmethod
    def categorical_crossentropy(outputs, y_true):
        loss = T.categorical_crossentropy(outputs['output'], y_true)
        return loss

    @staticmethod
    def mean_square_error(outputs, y_true):
        loss = T.sqr(outputs['output'] - y_true)
        return loss

    @staticmethod
    def mean_absolute_error(outputs, y_true):
        loss = T.abs(outputs['output'] - y_true)
        return loss

    @staticmethod
    def root_mean_square_error(outputs, y_true):
        loss = T.sqrt(T.sqr(outputs['output'] - y_true))
        return loss


class metrics:
    @staticmethod
    def commen(score, callback):
        score = callback(score)
        return score

    class Error:
        def __init__(self):
            self.name = 'Err'
            self.down_direction = True

        def evaluate(self, sample, y_true):
            y = sample['predict']
            score = T.neq(y, y_true)
            return score

    error = Error()

    class Accuracy:
        def __init__(self):
            self.name = 'Acc'
            self.down_direction = False

        def evaluate(self, sample, y_true):
            y = sample['predict']
            score = T.eq(y, y_true)
            return score

    accuracy = Accuracy()

    class Mean_Square_Error:
        def __init__(self):
            self.name = 'Mse'
            self.down_direction = True

        def evaluate(self, sample, y_true):
            y = sample['predict']
            return lossfunctions.mean_square_error({'output': y}, y_true)

    mean_square_error = Mean_Square_Error()

    class Mean_Absolute_Error:
        def __init__(self):
            self.name = 'Mae'
            self.down_direction = True

        def evaluate(self, sample, y_true):
            y = sample['predict']
            return lossfunctions.mean_absolute_error({'output': y}, y_true)

    mean_absolute_error = Mean_Absolute_Error()

    class Root_Mean_Square_Error:
        def __init__(self):
            self.name = 'Rmse'
            self.down_direction = True

        def evaluate(self, sample, y_true):
            y = sample['predict']
            return lossfunctions.root_mean_square_error({'output': y}, y_true)

    root_mean_square_error = Root_Mean_Square_Error()

    class True_Possitive:
        def __init__(self):
            self.name = 'Tp'
            self.down_direction = False

        def evaluate(self, sample, y_true):
            y = sample['predict']
            score = T.and_(y, y_true)
            return score

    true_possitive = True_Possitive()

    class False_Possitive:
        def __init__(self):
            self.name = 'Fp'
            self.down_direction = True

        def evaluate(self, sample, y_true):
            y = sample['predict']
            score = T.and_(T.eq(y, 1), T.not_(y_true))
            return score

    false_possitive = False_Possitive()

    class False_Negetive:
        def __init__(self):
            self.name = 'Fn'
            self.down_direction = True

        def evaluate(self, sample, y_true):
            y = sample['predict']
            score = T.mean(T.and_(T.eq(y, 0), y_true))
            return score

    false_negetive = False_Negetive()

    class Precision:
        def __init__(self):
            self.name = 'Pcs'
            self.down_direction = False

        def evaluate(self, sample, y_true):
            tp = metrics.true_possitive.evaluate(sample, y_true)
            fp = metrics.false_possitive.evaluate(sample, y_true)
            score = tp / (tp + fp)
            return score

    precision = Precision()

    class Recall:
        def __init__(self):
            self.name = 'Rcl'
            self.down_direction = False

        def evaluate(self, sample, y_true):
            tp = metrics.true_possitive.evaluate(sample, y_true)
            fn = metrics.false_negetive.evaluate(sample, y_true)
            score = tp / (tp + fn)
            return score

    recall = Recall()

    class GScore:
        def __init__(self):
            self.name = 'Gs'
            self.down_direction = False

        def evaluate(self, sample, y_true):
            tp = metrics.true_possitive.evaluate(sample, y_true)
            fp = metrics.false_possitive.evaluate(sample, y_true)
            fn = metrics.false_negetive.evaluate(sample, y_true)
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            score = T.sqrt(p * r)
            return score

    gscore = GScore()

    class FScore:
        def __init__(self, beta):
            self.beta = beta
            self.name = 'F{}'.format(str(beta))
            self.down_direction = False

        def evaluate(self, sample, y_true):
            tp = metrics.true_possitive.evaluate(sample, y_true)
            fp = metrics.false_possitive.evaluate(sample, y_true)
            fn = metrics.false_negetive.evaluate(sample, y_true)
            p = tp / (tp + fp)
            r = tp / (tp + fn)
            score = (1 + self.beta ** 2) * ((p * r) / ((self.beta ** 2) * p + r))
            return score

    f2 = FScore(2)
    f1 = FScore(1)
    fdot5 = FScore(0.5)


class Model(object):
    def __init__(self, X, Y, lossfunction=lossfunctions.categorical_crossentropy,
                 valid_lossfunction=None, metrics=(metrics.accuracy,)):
        self.init_data(X, Y)
        self.init_model()
        self.lossfunction = self.parse_lossfunction(lossfunction)
        if valid_lossfunction is None:
            self.valid_lossfunction = self.lossfunction
        else:
            self.valid_lossfunction = self.parse_lossfunction(valid_lossfunction)
        self.metrics = [self.parse_metric(m) for m in metrics]

    def init_model(self):
        self.output_layer = None
        self.output = None
        self.optimizer = None
        self.running_output = None
        self.outputs = OrderedDict()
        self.running_outputs = OrderedDict()
        self.loss = None
        self.sample = None
        self.sample_loss = None
        self.sample_score = None
        self.prediction = None
        self.params = OrderedDict()
        self.updates = OrderedDict()
        self.running_updates = OrderedDict()
        self.optimizer_updates = OrderedDict()
        self.train_updates = OrderedDict()
        self.user_debug_stream = []
        self.n_layer = 0
        self.layers = OrderedDict()
        self.datalayers = []
        self._betweenlayerops = OrderedDict()
        self._lossops = []
        self._train_fn = None
        self._valid_fn = None
        self._sample_fn = None
        self._predict_fn = None
        self._last_added_layer = self.input_layer
        self._built = False
        self._compiled = False

    def init_data(self, X, Y):
        self.X = X.X
        self.Y = Y.Y
        self.model_inputs = OrderedDict()
        self.model_outputs = OrderedDict()
        self.model_inputs['X'] = self.X
        self.model_outputs['Y'] = self.Y
        self.input_layer = Input(X)

    def parse_metric(self, metric):
        if metric in ['error']:
            metric = metrics.error
        if metric in ['accuracy']:
            metric = metrics.accuracy
        elif metric in ['mse', 'mean_square_error']:
            metric = metrics.mean_square_error
        elif metric in ['rmse', 'root_mean_square_error']:
            metric = metrics.root_mean_square_error
        elif metric in ['mae', 'mean_absolute_error']:
            metric = metrics.mean_absolute_error
        elif metric in ['tp', 'false_possitive']:
            metric = metrics.true_possitive
        elif metric in ['fp', 'false_possitive']:
            metric = metrics.false_possitive
        elif metric in ['fn', 'false_negetive']:
            metric = metrics.false_negetive
        elif metric in ['f1']:
            metric = metrics.f1
        elif metric in ['f2']:
            metric = metrics.f2
        elif metric in ['f0.5', 'f.5', 'fhalf', 'fdot5']:
            metric = metrics.fdot5
        elif metric in ['g', 'gscore']:
            metric = metrics.gscore
        return metric

    def parse_lossfunction(self, lossfunction):
        if lossfunction in ['ce', 'cce', 'categorical_cross_entropy', 'nlog', 'log_likelihood', 'neg_log']:
            lossfunction = lossfunctions.categorical_crossentropy
        if lossfunction in ['bce', 'cross_entropy', 'binary_cross_entropy']:
            lossfunction = lossfunctions.binary_crossentropy
        elif lossfunction in ['mse', 'mean_square_error']:
            lossfunction = lossfunction.mean_square_error
        elif lossfunction in ['rmse', 'root_mean_square_error']:
            lossfunction = lossfunction.root_mean_square_error
        elif lossfunction in ['mae', 'mean_absolute_error']:
            lossfunction = lossfunction.mean_absolute_error
        return lossfunction

    def optimize(self, optimizer):
        self.optimizer = optimizer
        if optimizer != None:
            params = OrderedDict()
            for name, param in self.params.items():
                if param.trainable:
                    params[name] = param()
            self.optimizer_updates = optimizer.build(params, self.loss)
        self.train_updates.update(self.updates)
        self.train_updates.update(self.optimizer_updates)

    def build(self):
        def output_callback(output):
            output = output
            for data_layer in self.datalayers:
                output = data_layer.get_output(output)
            return output

        def loss_callback(loss):
            loss = loss
            for data_layer in self.datalayers:
                loss = data_layer.get_loss(loss)
            return loss

        def output_callback_(output):
            output = output
            for data_layer in self.datalayers:
                output = data_layer.get_output_(output)
            return output

        def loss_callback_(loss):
            loss = loss
            for data_layer in self.datalayers:
                loss = data_layer.get_loss_(loss)
            return loss

        def score_callback(score):
            score = score
            for data_layer in self.datalayers:
                score = data_layer.get_score(score)
            return score

        def build_train():
            for name, layer in self.layers.items():
                layer.build_prepare()
                layer.build_train()
                if layer in self._betweenlayerops:
                    self._betweenlayerops[layer].op()
                self.updates.update(layer.updates)
            self.output_layer = self.layers.values()[-1]
            self.output = self.output_layer.output
            self.output = output_callback(self.output)
            self.outputs = self.output_layer.outputs
            self.loss = self.lossfunction(self.outputs, self.Y)
            self.loss = lossfunctions.commen(self.loss, loss_callback)
            for ops in self._lossops:
                self.loss = ops.build(self.loss)

        def build_running():
            for name, layer in self.layers.items():
                layer.build_running()
                if layer in self._betweenlayerops:
                    self._betweenlayerops[layer].op()
                self.params.update(layer.params)
                self.running_updates.update(layer.running_updates)
                self.user_debug_stream.extend(layer.debug_stream)
            self.output_layer = self.layers.values()[-1]
            self.running_output = self.output_layer.running_output
            self.running_output = output_callback_(self.running_output)
            self.running_outputs = self.output_layer.running_outputs
            self.sample = self.output_layer.sample(self.running_output)
            self.sample_loss = self.valid_lossfunction(self.running_outputs, self.Y)
            self.sample_loss = lossfunctions.commen(self.sample_loss, loss_callback_)
            self.sample_score = [metrics.commen(m.evaluate(self.sample, self.Y), score_callback) for m in self.metrics]
            self.prediction = self.output_layer.predict(self.running_output)

        if not self._built:
            build_train()
            build_running()

    def compile(self):
        def apply_train_fn():
            if T.is_graph(self.loss):
                train_fn = kernel.compile(inputs=self.model_inputs.values() + self.model_outputs.values(),
                                          outputs=[self.loss],
                                          updates=self.train_updates)
            else:
                train_fn = self.loss
            return train_fn

        def apply_valid_fn():
            valid_fn = kernel.compile(inputs=self.model_inputs.values() + self.model_outputs.values(),
                                      outputs=[self.sample_loss] + self.sample_score,
                                      updates=self.running_updates)
            return valid_fn

        def apply_sample_fn():
            sample_fn = kernel.compile(inputs=self.model_inputs.values() + self.model_outputs.values(),
                                       outputs=[self.sample_loss] + self.sample.values(),
                                       updates=self.train_updates)
            return sample_fn

        def apply_predict_fn():
            if T.is_graph(self.prediction):
                predict_fn = kernel.compile(inputs=self.model_inputs.values(),
                                            outputs=[self.prediction],
                                            updates=self.running_updates)
            else:
                predict_fn = self.prediction
            return predict_fn

        if not self._built:
            self.build()

        if self.optimizer is None:
            raise RuntimeError('Set the optimizer of Model first. Use .optimize()')

        if not self._compiled:
            logger("Compile", 0, 1)
            if config.is_log_detail():
                logger('Compiling Training Model', 1)
            train_fn = apply_train_fn()
            if config.is_log_detail():
                logger('Compiling Validation Model', 1)
            valid_fn = apply_valid_fn()
            if config.is_log_detail():
                logger('Compiling Sampling Model', 1)
            sample_fn = apply_sample_fn()
            if config.is_log_detail():
                logger('Compiling Predicting Model', 1)
            predict_fn = apply_predict_fn()
            self._train_fn = train_fn
            self._valid_fn = valid_fn
            self._sample_fn = sample_fn
            self._predict_fn = predict_fn

    def prepare_data(self, data, running=True):
        data_x, data_y = data
        x = copy.deepcopy(data_x)
        y = copy.deepcopy(data_y)
        data = OrderedDict()
        data['X'] = x
        data['Y'] = y
        for dl in self.datalayers:
            if not running:
                dl.apply(data)
            else:
                dl.apply_(data)
        data_list = []
        for name in self.model_inputs:
            data_list.append(data[name])
        for name in self.model_outputs:
            data_list.append(data[name])
        return tuple(data_list)

    def fit(self, data):
        return MainLoop.train(data, self, self.optimizer, extensions=[extensions.monitor, extensions.earlystop])

    def train(self, X, Y):
        data = self.prepare_data([X, Y], running=False)
        return np.mean(self._train_fn(*data))

    def valid(self, X, Y):
        data = self.prepare_data([X, Y])
        valid_result = self._valid_fn(*data)
        result = {'loss': valid_result[0]}
        for i, m in enumerate(self.metrics):
            result[m.name] = valid_result[i + 1]
        return result

    def sampler(self, X, Y):
        data = self.prepare_data([X, Y])
        sample_lists = self._sample_fn(*data)
        sample_results = OrderedDict()
        sample_results['sample_loss'] = sample_lists[0]
        for i, key in enumerate(self.sample):
            sample_results[key] = sample_lists[i+1]
        return sample_results

    def predict(self, X, Y):
        data = self.prepare_data([X, Y])
        return self._predict_fn(*data)

    def save(self, filepath):
        extensions.SaveLoad.save_file(self, filepath)

    def load(self, filepath):
        extensions.SaveLoad.load_params(self, filepath)

    def add(self, element, name=None):
        if isinstance(element, LayerBase):
            self.n_layer += 1
            if name == None: name = 'layer{}'.format(self.n_layer)
            self.layers[name] = element
            element.initiate(name, self._last_added_layer, self.model_inputs, self.model_outputs)
            self._last_added_layer = element

        elif isinstance(element, DataBase):
            element.set_model(self)
            self.datalayers.append(element)

        elif isinstance(element, Inlayerops):
            self._last_added_layer.update_ops(element)

        elif isinstance(element, Betweenlayerops):
            self._betweenlayerops[self._last_added_layer](element)
            element.init(self._last_added_layer)

        elif isinstance(element, Lossops):
            self._lossops.append(element)
            element.init(self._last_added_layer, self)
