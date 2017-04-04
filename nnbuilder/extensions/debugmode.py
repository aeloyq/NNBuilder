# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import extension
import sys
sys.path.append('..')
import theano
from collections import OrderedDict
import numpy as np
base=extension.extension

class ex(base):
    def __init__(self,kwargs):
        base.__init__(self,kwargs)
        self.kwargs=kwargs
        self.debug_batch=3
        self.debug_time=1
    def init(self):
        base.init(self)
    def before_train(self):
        kwargs=self.kwargs
        self.logger('Compiling Debug Model', 1)
        model=kwargs['dim_model']
        self.inputs=model.inputs
        values = []
        user_values = []
        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['data_stream']
        for time in range(self.debug_time):
            data = kwargs['prepare_data'](train_X, train_Y, (np.arange(self.debug_batch)+self.debug_batch*time).tolist())
            data = tuple(data)
            self.logger("\r\n\r\nInput Debug Info:\r\n\r\n", 1)
            self.logger('%sth input debug' % (time + 1), 2, 1)
            self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
            for d, inp in zip(data, model.inputs):
                self.logger('input:' + str(inp) + ':' + str(np.array(d).shape), 3)
                self.logger(str(np.array(d)) + '    id : %s' % len(values), 4)
                values.append(d)
                self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
            self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)


            self.logger("\r\n\r\nUser Debug Info:\r\n\r\n", 1)
            self.logger('%sth user debug' % (time + 1), 2, 1)
            self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 2)
            for ud in model.user_debug_stream:
                fn = self.get_func(ud)
                user_debug_result = fn(*data)
                self.logger('name : ' + str(ud)+'  with shape:%s'%str(user_debug_result.shape), 2)
                self.logger(str(user_debug_result) + '    id : %s' % len(user_values), 3)
                user_values.append(user_debug_result)
                self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 2)
            self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)


            self.logger("\r\n\r\nModel Debug Info:\r\n\r\n", 1)
            self.logger('%sth model_debug' % (time + 1), 2, 1)
            for key,layer in model.layers.items():
                debug_stream=[layer.input,layer.output]
                fn=self.get_func(debug_stream)
                debug_result=fn(*data)
                self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)

                self.logger(layer.name + ':', 2, 1)
                self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
                self.logger(
                    'input:    %s' % layer.input + '  shape:' + str(debug_result[0].shape), 3)
                self.logger('pprint:', 3)
                self.logger(str(theano.pp(layer.input)), 3)
                self.logger('value:', 3)
                self.logger(str(debug_result[0]) + '    id : %s' % len(values), 4)
                values.append(debug_result[0])

                self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
                self.logger('params' + ':', 3)
                for param in layer.params:
                    self.logger('%s' % param + ' : ' + str(param.get_value().shape), 3)
                    self.logger(str(param.get_value()) + '    id : %s' % len(values), 4)
                    values.append(param.get_value())
                self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)

                self.logger(
                    'output:    %s' % layer.output + '  shape:' + str(debug_result[1].shape), 3)
                self.logger('pprint:', 3)
                self.logger('▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲', 3)
                self.logger(str(theano.pp(layer.output)), 3)
                self.logger('▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲', 3)
                self.logger('value:', 3)
                self.logger(str(debug_result[1]) + '    id : %s' % len(values), 4)
                values.append(debug_result[1])
                self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)


            self.logger("\r\n\r\nOutput Debug Info:\r\n\r\n", 1)
            debug_stream=[model.output.output,model.pred_Y,model.cost,model.error]
            fn = self.get_func(debug_stream)
            debug_result = fn(*data)
            self.logger('%sth output' % (time + 1), 2, 1)
            self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)
            self.logger('Output : ' + str(debug_result[0].shape), 2)
            self.logger(str(debug_result[0]) + '    id : %s' % len(values), 3)
            values.append(debug_result[0])
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
            self.logger('Predict : ' + str(debug_result[1].shape), 2)
            self.logger(str(debug_result[1]) + '    id : %s' % len(values), 3)
            values.append(debug_result[1])
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
            self.logger('Cost : ' + str(debug_result[2].shape), 2)
            self.logger(str(debug_result[2]) + '    id : %s' % len(values), 3)
            values.append(debug_result[2])
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
            self.logger('Error : ' + str(debug_result[3].shape), 2)
            self.logger(str(debug_result[3]) + '    id : %s' % len(values), 3)
            values.append(debug_result[3])
            self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)

            self.logger("\r\n\r\nUpdate Debug Info:\r\n\r\n", 1)
            debug_stream=[]
            up=[]
            for update in model.updates:
                debug_stream.append(update[1])
                up.append(update[0])

            fn = self.get_up_func(debug_stream, model.updates)
            debug_result = fn(*data)
            self.logger('%sth update'%(time+1), 2, 1)
            self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
            for i,u in enumerate(up):
                up_shape='Not Shown'
                try:
                    up_shape=u.get_value().shape
                except:
                    pass
                self.logger('Variable to update : ' + str(u)+'  with shape: %s'%str(up_shape), 2)
                self.logger('Update values with shape: ' + str(debug_result[i].shape), 2)
                self.logger(str(debug_result[i]) + '    id : %s' % len(values), 3)
                values.append(debug_result[i])
                self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
            self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)


        kwargs['debug_result'] = [values,user_values]
        kwargs['stop'] = True


    def get_func(self,output):
        debug_model = theano.function(inputs=self.inputs,
                                      outputs=output,
                                      on_unused_input='ignore')
        return  debug_model
    def get_up_func(self,output,update):
        debug_model = theano.function(inputs=self.inputs,
                                      outputs=output,updates=update,
                                      on_unused_input='ignore')
        return  debug_model

config=ex({})