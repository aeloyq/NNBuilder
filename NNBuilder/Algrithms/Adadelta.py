# -*- coding: utf-8 -*-
"""
Created on  Feb 14 2:02 AM 2017

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
        self.p_grads=[]
        self.pgrads=[]
        self.fn_inputs=self.gparams
        tensor_var_dict = {'1': T.vector, '2': T.matrix, '3': T.ftensor3}
        for param in self.params:
            i = tensor_var_dict['%d' % param.ndim]()
            self.fn_inputs.append(i)
            self.pgrads.append(i)
            value = param.get_value()
            self.p_grads.append(np.zeros_like(value))
    def get_updates(self):
        self.updates2output = [
            (1 / (T.sqrt(T.square((1 - 0.95) * p_grad) + T.square(0.95 * gparam) + 1.e-6))) * gparam for
            gparam, p_grad in zip(self.gparams, self.pgrads)]
        self.updates=[(param, param - updts2otpt)
               for param, updts2otpt in zip(self.params, self.updates2output)]
        return self.updates
    def get_update_func(self):
        fn=theano.function(inputs=self.fn_inputs,outputs=self.updates2output,updates=self.updates)
        return fn
    def repeat(self,argv,argv2):
        self.save_p_grad(argv)
    def save_p_grad(self, p_grads):
        self.p_grads=[np.array(p_grad) for p_grad in p_grads]
    def get_input(self,grads):
        argv=[]
        for grad in grads:
            argv.append(grad)
        argv.extend(self.p_grads)
        return tuple(argv)