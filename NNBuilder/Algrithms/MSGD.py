# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:05 PM 2017

@author: aeloyq
"""

import theano
import theano.tensor as T

class algrithm:
    def __init__(self,configuration,wt_packs):
        self.params=[theta for param in wt_packs for theta in param]
        self.configuration=configuration
        self.gparams = []
        tensor_var_dict = {'1': T.vector, '2': T.matrix, '3': T.ftensor3}
        for param in self.params:
            self.gparams.append(tensor_var_dict['%d' % param.ndim]())
    def set_params(self,wt_packs):
        self.params = [theta for param in wt_packs for theta in param]
    def get_updates(self):
        updates=[(param, param - self.configuration['learning_rate'] * gparam)
               for param, gparam in zip(self.params, self.gparams)]
        return updates
    def get_update_func(self):
        fn=theano.function(inputs=self.gparams,updates=self.get_updates())
        return fn
    def repeat(self,argv):
        pass
    def save_p_grad(self,p_grad):
        pass
    def get_input(self,grads):
        return tuple(grads)