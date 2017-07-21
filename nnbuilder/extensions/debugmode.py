# -*- coding: utf-8 -*-
"""
Created on  Feb 14 1:22 PM 2017

@author: aeloyq
"""
import theano
import numpy as np
from collections import OrderedDict
from nnbuilder.main import mainloop
from extension import extension

class ex(extension):
    def __init__(self,kwargs):
        extension.__init__(self,kwargs)
        self.kwargs=kwargs
        self.debug_batch=3
        self.debug_time=1
        self.start_index=0
    def init(self):
        extension.init(self)
    def debug(self):
        kwargs=self.kwargs
        self.logger('Compiling Debug Model', 0)
        model=kwargs['model']
        self.inputs=model.inputs
        self.updates=model.updates
        values = []
        user_values = []
        X,  Y, = kwargs['data']
        for time in range(self.debug_time):
            index=np.arange(self.debug_batch)+self.debug_batch*time+self.start_index
            data = mainloop.prepare_data(X, Y, (index).tolist())
            data = tuple(data)
            self.logger("\r\n\r\nInput Debug Info:\r\n\r\n", 0)
            self.logger('%sth input debug' % (time + 1), 1, 1)
            self.logger('index from %s to %s' % (index[0],index[-1]), 1, 1)
            self.logger('=============================================================', 1, 1)
            self.logger('-----------------------------------------------', 2)
            for d, inp in zip(data, model.inputs):
                self.logger('input:' + str(inp) + ':' + str(np.array(d).shape), 2)
                self.logger(str(np.array(d)) + '    id : %s' % len(values), 3)
                values.append(d)
                self.logger('-----------------------------------------------', 2)
            self.logger('=============================================================', 1, 1)


            self.logger("\r\n\r\nUser Debug Info:\r\n\r\n", 1)
            self.logger('%sth user debug' % (time + 1), 2, 1)
            self.logger('=============================================================', 1, 1)
            self.logger('-----------------------------------------------', 1)
            for ud in model.user_debug_stream:
                fn = self.get_up_func(ud,self.updates)
                user_debug_result = fn(*data)
                self.logger('name : ' + str(ud)+'  with shape:%s'%str(user_debug_result.shape), 1)
                self.logger(str(user_debug_result) + '    id : %s' % len(user_values), 2)
                user_values.append(user_debug_result)
                self.logger('-----------------------------------------------', 1)
            self.logger('=============================================================', 1, 1)


            self.logger("\r\n\r\nModel Debug Info:\r\n\r\n", 0)
            self.logger('%sth model_debug' % (time + 1), 2, 1)
            for key,layer in model.layers.items():
                debug_stream=[layer.input,layer.output]
                fn = self.get_up_func(debug_stream, self.updates)
                debug_result=fn(*data)
                self.logger('=============================================================', 1, 1)

                self.logger(layer.name + ':', 2, 1)
                self.logger('-----------------------------------------------', 2)
                self.logger(
                    'input:    %s' % layer.input + '  shape:' + str(debug_result[0].shape), 2)
                self.logger('pprint:', 3)
                self.logger(str(theano.pp(layer.input))[:100], 2)
                self.logger('value:', 3)
                self.logger(str(debug_result[0]) + '    id : %s' % len(values), 3)
                values.append(debug_result[0])

                self.logger('-----------------------------------------------', 2)
                self.logger('params' + ':', 3)
                for name,param in layer.params.items():
                    self.logger('%s' % param + ' : ' + str(param.get_value().shape), 2)
                    self.logger(str(param.get_value()) + '    id : %s' % len(values), 3)
                    values.append(param.get_value())
                self.logger('-----------------------------------------------', 2)

                self.logger(
                    'output:    %s' % layer.output + '  shape:' + str(debug_result[1].shape), 2)
                self.logger('value:', 3)
                self.logger(str(debug_result[1]) + '    id : %s' % len(values), 3)
                values.append(debug_result[1])
                self.logger('-----------------------------------------------', 2)


            self.logger("\r\n\r\nOutput Debug Info:\r\n\r\n", 0)
            debug_stream=[model.output.output,model.predict,model.cost,model.raw_cost,model.error]
            updt=OrderedDict()
            updt.update(self.updates)
            updt.update(model.raw_updates)
            fn = self.get_up_func(debug_stream,updt)
            debug_result = fn(*data)
            self.logger('%sth output' % (time + 1), 2, 1)
            self.logger('=============================================================', 1, 1)
            self.logger('Output : ' + str(debug_result[0].shape), 2)
            self.logger(str(debug_result[0]) + '    id : %s' % len(values), 2)
            values.append(debug_result[0])
            self.logger('-----------------------------------------------', 2)
            self.logger('Predict : ' + str(debug_result[1].shape), 2)
            self.logger(str(debug_result[1]) + '    id : %s' % len(values), 2)
            values.append(debug_result[1])
            self.logger('-----------------------------------------------', 2)
            self.logger('Cost : ' + str(debug_result[2].shape), 2)
            self.logger(str(debug_result[2]) + '    id : %s' % len(values), 2)
            values.append(debug_result[2])
            self.logger('-----------------------------------------------', 2)
            self.logger('Raw Cost : ' + str(debug_result[2].shape), 2)
            self.logger(str(debug_result[3]) + '    id : %s' % len(values), 2)
            values.append(debug_result[3])
            self.logger('-----------------------------------------------', 2)
            self.logger('Error : ' + str(debug_result[3].shape), 2)
            self.logger(str(debug_result[4]) + '    id : %s' % len(values), 2)
            values.append(debug_result[4])
            self.logger('=============================================================', 1, 1)

            self.logger("nUpdate Debug Info:", 0,2)
            debug_stream=[]
            up=[]
            for name,update in model.optimizer_updates.items():
                debug_stream.append(update)
                up.append(name)

            fn = self.get_up_func(debug_stream, updt)
            debug_result = fn(*data)
            self.logger('%sth update'%(time+1), 2, 1)
            self.logger('=============================================================', 1, 1)
            self.logger('-----------------------------------------------', 2)
            for i,u in enumerate(up):
                up_shape='Not Shown'
                try:
                    up_shape=u.get_value().shape
                except:
                    pass
                self.logger('Variable to update : ' + str(u)+'  with shape: %s'%str(up_shape), 1)
                self.logger('Update values with shape: ' + str(debug_result[i].shape), 1)
                self.logger(str(debug_result[i]) + '    id : %s' % len(values), 2)
                values.append(debug_result[i])
                self.logger('-----------------------------------------------', 2)
            self.logger('=============================================================', 1, 1)


        kwargs['debug_result'] = [values,user_values]


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