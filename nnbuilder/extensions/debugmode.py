# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""

import numpy as np
from collections import OrderedDict
from nnbuilder.main import mainloop
from basic import base
from nnbuilder.kernel import *


class ex(base):
    def __init__(self, kwargs):
        base.__init__(self, kwargs)
        self.kwargs = kwargs
        self.debug_batch = 3
        self.debug_time = 1
        self.start_index = 0

    def init(self):
        base.init(self)

    def set_logattr(self):
        self.logattr = ['debug_batch', 'debug_time', 'start_index']

    def debug(self):
        kwargs = self.kwargs
        self.logger('Compiling Debug Model', 0)
        model = kwargs['model']
        self.inputs = model.inputs
        self.updates = model.updates
        values = []
        user_values = []
        X, Y = kwargs['data']
        for time in range(self.debug_time):
            index = np.arange(self.debug_batch) + self.debug_batch * time + self.start_index
            data = mainloop.prepare_data(X, Y, (index).tolist(), model)
            data = tuple(data)
            self.logger("\r\n\r\nInput Debug Info:\r\n\r\n", 0)
            self.logger('%sth input debug' % (time + 1), 1, 1)
            self.logger('index from %s to %s' % (index[0], index[-1]), 1, 1)
            self.logger('=============================================================', 1, 1)
            self.logger('-----------------------------------------------', 1)
            for d, inp in zip(data, model.inputs):
                self.logger(
                    'input:' + str(inp) + ':' + str(np.array(d).shape),
                    1)
                self.logger(str(np.array(d)) + '    id : %s' % len(values), 2)
                values.append(d)
                self.logger('-----------------------------------------------', 1)
            self.logger('=============================================================', 1, 1)

            self.logger("\r\n\r\nUser Debug Info:\r\n\r\n", 0)
            self.logger('%sth user debug' % (time + 1), 1, 1)
            self.logger('=============================================================', 1, 1)
            self.logger('-----------------------------------------------', 1)
            for ud in model.user_debug_stream:
                fn = self.get_up_func(ud, self.updates)
                user_debug_result = fn(*data)
                self.logger('name : ' + str(ud) + '  with shape:%s' % str(user_debug_result.shape), 1)
                self.logger(str(user_debug_result) + '    id : %s' % len(user_values), 2)
                user_values.append(user_debug_result)
                self.logger('-----------------------------------------------', 1)
            self.logger('=============================================================', 1, 1)

            self.logger("\r\n\r\nModel Debug Info:\r\n\r\n", 0)
            self.logger('%sth model_debug' % (time + 1), 1, 1)
            for key, layer in model.layers.items():
                debug_stream = [layer.input, layer.output]
                fn = self.get_up_func(debug_stream, self.updates)
                debug_result = fn(*data)
                self.logger('=============================================================', 1, 1)

                self.logger(layer.name + ':', 1, 1)
                self.logger('-----------------------------------------------', 1)
                self.logger(
                    'input : %s' % str(debug_result[0].shape), 1)
                self.logger(str(debug_result[0]) + '    id : %s' % len(values), 2)
                values.append(debug_result[0])

                self.logger('-----------------------------------------------', 1)
                self.logger('params' + ':', 1, 1)
                for name, param in layer.params.items():
                    self.logger('%s' % param + ' : ' + str(param.get().shape), 1)
                    self.logger(str(param.get()) + '    id : %s' % len(values), 2)
                    values.append(param.get())
                self.logger('-----------------------------------------------', 1)

                self.logger('untrainable params' + ':', 1, 1)
                for name, oparam in layer.untrainable_params.items():
                    self.logger('%s' % oparam + ' : ' + str(oparam.get().shape), 1)
                    self.logger(str(oparam.get()) + '    id : %s' % len(values), 2)
                    values.append(oparam.get())
                self.logger('-----------------------------------------------', 1)

                self.logger(
                    'output : %s' % str(debug_result[1].shape), 1)
                self.logger(str(debug_result[1]) + '    id : %s' % len(values), 2)
                values.append(debug_result[1])
                self.logger('-----------------------------------------------', 1)

            self.logger("\r\n\r\nOutput Debug Info:\r\n\r\n", 0)
            debug_stream = [model.output, model.raw_output, model.loss, model.sample, model.sample_loss,
                            model.sample_error, model.predict]
            updt = OrderedDict()
            updt.update(self.updates)
            updt.update(model.raw_updates)
            fn = self.get_up_func(debug_stream, updt)
            debug_result = fn(*data)
            self.logger('%sth output' % (time + 1), 1, 1)
            self.logger('=============================================================', 1, 1)
            self.logger('Output : ' + str(debug_result[0].shape), 1)
            self.logger(str(debug_result[0]) + '    id : %s' % len(values), 2)
            values.append(debug_result[0])
            self.logger('-----------------------------------------------', 1)
            self.logger('Raw Output : ' + str(debug_result[0].shape), 1)
            self.logger(str(debug_result[1]) + '    id : %s' % len(values), 2)
            values.append(debug_result[1])
            self.logger('-----------------------------------------------', 1)
            self.logger('Loss : ' + str(debug_result[2].shape), 1)
            self.logger(str(debug_result[2]) + '    id : %s' % len(values), 2)
            values.append(debug_result[2])
            self.logger('-----------------------------------------------', 1)
            self.logger('Sample : ' + str(debug_result[1].shape), 1)
            self.logger(str(debug_result[3]) + '    id : %s' % len(values), 2)
            values.append(debug_result[3])
            self.logger('-----------------------------------------------', 1)
            self.logger('Sample Loss : ' + str(debug_result[2].shape), 1)
            self.logger(str(debug_result[4]) + '    id : %s' % len(values), 2)
            values.append(debug_result[4])
            self.logger('-----------------------------------------------', 1)
            self.logger('Sample Error : ' + str(debug_result[3].shape), 1)
            self.logger(str(debug_result[5]) + '    id : %s' % len(values), 2)
            values.append(debug_result[5])
            self.logger('-----------------------------------------------', 1)
            self.logger('Predict : ' + str(debug_result[3].shape), 1)
            self.logger(str(debug_result[6]) + '    id : %s' % len(values), 2)
            values.append(debug_result[6])
            self.logger('=============================================================', 1, 1)

            self.logger("Update Debug Info:", 0, 2)
            debug_stream = []
            up = []
            for name, update in model.optimizer_updates.items():
                debug_stream.append(update)
                up.append(name)

            fn = self.get_up_func(debug_stream, updt)
            debug_result = fn(*data)
            self.logger('%sth update' % (time + 1), 1, 1)
            self.logger('=============================================================', 1, 1)
            self.logger('-----------------------------------------------', 2)
            for i, u in enumerate(up):
                up_shape = 'Not Shown'
                try:
                    up_shape = u.get().shape
                except:
                    pass
                self.logger(str(u) + '  with shape: %s' % str(up_shape), 1)
                self.logger(str(debug_result[i]) + '    id : %s' % len(values), 2)
                values.append(debug_result[i])
                self.logger('-----------------------------------------------', 2)
            self.logger('=============================================================', 1, 1)

        kwargs['debug_result'] = [values, user_values]

    def get_func(self, output):
        debug_model = kernel.compile(inputs=self.inputs,
                                     outputs=output,
                                     strict=False)
        return debug_model

    def get_up_func(self, output, update):
        if not isinstance(output,list):
            output=[output]
        debug_model = kernel.compile(inputs=self.inputs,
                                     outputs=output, updates=update,
                                     strict=False)
        return debug_model


config = ex({})
instance = config
