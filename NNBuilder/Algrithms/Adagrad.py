# -*- coding: utf-8 -*-
"""
Created on  Feb 14 4:37 PM 2017

@author: aeloyq
"""

import theano
import theano.tensor as T
import numpy as np
import MSGD

base=MSGD.algrithm
class algrithm(base):
    def __init__(self,configuration,wt_packs):
        base.__init__(self,configuration,wt_packs)
        self.pregularizers=[]
        self.p_regularizers=[]
        self.fn_inputs=self.gparams
        tensor_var_dict={'1':T.vector,'2':T.matrix,'3':T.ftensor3}
        for param in self.params:
            i = tensor_var_dict['%d'%param.ndim]()
            self.fn_inputs.append(i)
            self.pregularizers.append(i)
            value=param.get_value()
            self.p_regularizers.append(np.ones_like(value)/1.e6)
    def get_updates(self):
        updates=[(param, param - (self.configuration['learning_rate'] / T.sqrt(pregularizer)) * gparam)
               for param, gparam, pregularizer in zip(self.params, self.gparams,self.pregularizers)]
        return updates
    def get_update_func(self):
        fn=theano.function(inputs=self.fn_inputs,updates=self.get_updates())
        return fn
    def repeat(self,argv):
        self.save_p_regularizer(argv)
    def save_p_regularizer(self, regs):
        for p_regularizer,reg in zip(self.p_regularizers,regs):
            p_regularizer+=np.square(reg)
    def get_input(self,grads):
        argv = []
        for grad in grads:
            argv.append(grad)
        argv.extend(self.p_regularizers)
        return tuple(argv)