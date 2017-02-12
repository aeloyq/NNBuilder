# -*- coding: utf-8 -*-
"""
Created on Sun Dec 18 00:48:18 2016

@author: aeloyq
"""

import theano.tensor as T
import numpy as np
import NNBuilder as nnb

''' configurations '''

rng = np.random.RandomState(1234)


def get_conf_xor():
    configuration = {}
    rng = np.random.RandomState(1234)
    configuration['rng'] = rng
    # Data Preparations
    configuration['n_data'] = 100
    configuration['data_pre'] = nnb.Preparation.DataPrepares.Load_xor
    # Model Structs
    configuration['n_inputs'] = 2
    configuration['n_outputs'] = 1
    configuration['model_struct'] = [
        [[nnb.Layers.HiddenLayersFF.Hidden_Layer_FeedForward, (rng, 2, 2)], 'hidden1', 'raw', 2]]
    configuration['model_output'] = nnb.Layers.Logistic.MLP_Logistic
    configuration['Wt_init'] = 'uniform'
    configuration['Bi_init'] = 'zeros'
    configuration['cost_func'] = 'square'
    # Regularization Items
    configuration['L0_reg'] = 0.
    configuration['L1_reg'] = 0.
    configuration['L2_reg'] = 0.
    # MBGD Settings
    configuration['max_epoches'] = 1000
    configuration['learning_rate'] = 5
    configuration['batch_size'] = 5
    # Early Stop Settings
    configuration['is_early_stop'] = True
    configuration['patience_increase'] = 2
    configuration['train_patience'] = 10000
    configuration['improvement_threshold'] = 0.995
    configuration['valid_frequence'] = min(configuration['batch_size'], configuration['train_patience'] // 2)
    # Sample Settings
    configuration['sample_frequence'] = 40
    configuration['n_sample'] = 1
    configuration['sample_func'] = None
    # Log Settings
    configuration['report_per_literation'] = False
    configuration['report_per_epoch'] = False
    return configuration


def get_conf_xor_sm():
    configuration = {}
    # Data Preparations
    configuration['n_data'] = 100
    configuration['data_pre'] = nnb.Preparation.DataPrepares.Load_xor
    # Model Structs
    configuration['n_inputs'] = 2
    configuration['n_outputs'] = 1
    configuration['n_hidden'] = 2
    configuration['n_layers'] = 1
    configuration['hidden_activation'] = T.nnet.sigmoid
    configuration['model_struct'] = nnb.Layers.Softmax.MLP_Softmax
    # Regularization Items
    configuration['L0_reg'] = 0.
    configuration['L1_reg'] = 0.
    configuration['L2_reg'] = 0.
    # MBGD Settings
    configuration['max_epoches'] = 1000
    configuration['learning_rate'] = 1
    configuration['batch_size'] = 5
    configuration['sample_frequence'] = 40
    configuration['n_sample'] = 1
    configuration['train_patience'] = 10000
    configuration['valid_frequence'] = min(configuration['batch_size'], configuration['train_patience'] // 2)
    configuration['improvement_threshold'] = 0.995
    configuration['patience_increase'] = 2
    configuration['report_per_literation'] = False
    configuration['report_per_epoch'] = True
    configuration['sample_func'] = nnb.Extensions.Samples.xor_sample
    return configuration


def get_conf_mnist():
    configuration = {}
    # Data Preparations
    configuration['mnist_path'] = "./datasets/mnist.pkl.gz"
    configuration['data_pre'] = nnb.Preparation.DataPrepares.Load_mnist
    # Model Structs
    configuration['n_inputs'] = 28 * 28
    configuration['n_outputs'] = 10
    configuration['n_hidden'] = 500
    configuration['n_layers'] = 1
    configuration['hidden_activation'] = T.tanh
    configuration['model_struct'] = nnb.Layers.MLP_Softmax
    # Regularization Items
    configuration['L0_reg'] = 0.
    configuration['L1_reg'] = 0.
    configuration['L2_reg'] = 0.0001
    # MBGD Settings
    configuration['max_epoches'] = 1000
    configuration['learning_rate'] = 0.2
    configuration['batch_size'] = 20
    configuration['sample_frequence'] = -1
    configuration['n_sample'] = 2
    configuration['train_patience'] = 10000
    configuration['valid_frequence'] = 2500
    configuration['improvement_threshold'] = 0.995
    configuration['patience_increase'] = 2
    configuration['report_per_literation'] = False
    configuration['report_per_epoch'] = True
    configuration['sample_func'] = nnb.Extensions.Samples.mnist_sample
    return configuration
