# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""

from basic import *


class Debug(ExtensionBase):
    def __init__(self):
        ExtensionBase.__init__(self)
        self.name = 'debug'
        self.size = 3
        self.time = 1
        self.index = 0

    def debug(self):
        self.logger('Compiling Debug Model', 0)
        data = self.data
        model = self.model
        inputs = self.model.model_inputs.values() + self.model.model_outputs.values()
        results = []
        user_results = []
        for time in range(self.time):
            minibatch = np.arange(self.size) + self.size * time + self.index
            data = model.prepare_data(data.get_minibatch_train(minibatch))
            self.logger("\r\n\r\nInput Debug Info:\r\n\r\n", 0)
            self.logger('%sth input debug' % (time + 1), 1, 1)
            self.logger('index from %s to %s' % (minibatch[0], minibatch[-1]), 1, 1)
            self.logger('=============================================================', 1, 1)
            self.logger('-----------------------------------------------', 1)
            for d, inp in zip(data, model.model_inputs):
                self.logger(
                    'input:' + str(inp) + ':' + str(np.array(d).shape),
                    1)
                self.logger(str(np.array(d)) + '    id : %s' % len(results), 2)
                results.append(d)
                self.logger('-----------------------------------------------', 1)
            self.logger('=============================================================', 1, 1)

            self.logger("\r\n\r\nUser Debug Info:\r\n\r\n", 0)
            self.logger('%sth user debug' % (time + 1), 1, 1)
            self.logger('=============================================================', 1, 1)
            self.logger('-----------------------------------------------', 1)
            for ud in model.user_debug_stream:
                fn = self.get_fn(inputs, [ud])
                user_debug_result = fn(*data)
                self.logger('name : ' + str(ud) + '  with shape:%s' % str(user_debug_result.shape), 1)
                self.logger(str(user_debug_result) + '    id : %s' % len(user_results), 2)
                user_results.append(user_debug_result)
                self.logger('-----------------------------------------------', 1)
            self.logger('=============================================================', 1, 1)

            self.logger("\r\n\r\nModel Debug Info:\r\n\r\n", 0)
            self.logger('%sth model_debug' % (time + 1), 1, 1)
            for key, layer in model.layers.items():
                fn = self.get_fn(inputs, layer.output)
                debug_result = fn(*data)
                self.logger('=============================================================', 1, 1)

                self.logger(layer.name + ':', 1, 1)
                self.logger('-----------------------------------------------', 1)
                self.logger(
                    'input : %s' % str(debug_result[0].shape), 1)
                self.logger(str(debug_result[0]) + '    id : %s' % len(results), 2)
                results.append(debug_result[0])

                self.logger('-----------------------------------------------', 1)
                self.logger('params' + ':', 1, 1)
                for name, param in layer.params.items():
                    self.logger('%s' % param + ' : ' + str(param.get().shape), 1)
                    self.logger(str(param.get()) + '    id : %s' % len(results), 2)
                    results.append(param.get())
                self.logger('-----------------------------------------------', 1)

                self.logger('untrainable params' + ':', 1, 1)
                for name, oparam in layer.untrainable_params.items():
                    self.logger('%s' % oparam + ' : ' + str(oparam.get().shape), 1)
                    self.logger(str(oparam.get()) + '    id : %s' % len(results), 2)
                    results.append(oparam.get())
                self.logger('-----------------------------------------------', 1)

                self.logger(
                    'output : %s' % str(debug_result[1].shape), 1)
                self.logger(str(debug_result[1]) + '    id : %s' % len(results), 2)
                results.append(debug_result[1])
                self.logger('-----------------------------------------------', 1)

            self.logger("\r\n\r\nOutput Debug Info:\r\n\r\n", 0)
            debug_stream = [model.output, model.running_output]
            debug_stream += model.sample.values()
            debug_stream += [model.prediction, model.loss, model.sample_loss]
            debug_stream += model.sample_score
            fn = self.get_fn(inputs, debug_stream)
            debug_result = fn(*data)
            i = 0
            self.logger('%sth output' % (time + 1), 1, 1)
            self.logger('=============================================================', 1, 1)
            self.logger('Output : ' + str(debug_result[i].shape), 1)
            self.logger(str(debug_result[i]) + '    id : %s' % len(results), 2)
            results.append(debug_result[i])
            i += 1
            self.logger('-----------------------------------------------', 1)
            self.logger('Running Output : ' + str(debug_result[i].shape), 1)
            self.logger(str(debug_result[i]) + '    id : %s' % len(results), 2)
            results.append(debug_result[i])
            i += 1
            self.logger('-----------------------------------------------', 1)
            for name in model.sample:
                self.logger('Sample %s : ' % (name) + str(debug_result[i].shape), 1)
                self.logger(str(debug_result[i]) + '    id : %s' % len(results), 2)
                results.append(debug_result[i])
                i += 1
            self.logger('-----------------------------------------------', 1)
            self.logger('Predict : ' + str(debug_result[i].shape), 1)
            self.logger(str(debug_result[i]) + '    id : %s' % len(results), 2)
            results.append(debug_result[i])
            i += 1
            self.logger('-----------------------------------------------', 1)
            self.logger('Loss : ' + str(debug_result[i]) + '    id : %s' % len(results), 1)
            results.append(debug_result[i])
            i += 1
            self.logger('-----------------------------------------------', 1)
            self.logger('Sample Loss : ' + str(debug_result[i]) + '    id : %s' % len(results), 1)
            results.append(debug_result[i])
            i += 1
            self.logger('-----------------------------------------------', 1)
            for metric in model.metrics:
                self.logger('Sample %s : ' % (metric.name) + str(debug_result[5]) + '    id : %s' % len(results), 1)
                results.append(debug_result[i])
                i += 1
            self.logger('=============================================================', 1, 1)

            self.logger("Update Debug Info:", 0, 2)
            debug_stream = []
            up = []
            for name, update in model.optimizer_updates.items():
                debug_stream.append(update)
                up.append(name)

            fn = self.get_fn(inputs, debug_stream)
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
                self.logger(str(debug_result[i]) + '    id : %s' % len(results), 2)
                results.append(debug_result[i])
                self.logger('-----------------------------------------------', 2)
            self.logger('=============================================================', 1, 1)

        return {'results': results, 'user_results': user_results}

    def get_fn(self, input, output, update=None):
        return kernel.compile(inputs=input, outputs=output, updates=update, strict=False)


debug = Debug()
