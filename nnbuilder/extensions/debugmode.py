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
    def init(self):
        base.init(self)
    def before_train(self):
        kwargs=self.kwargs
        self.logger('Compiling Debug Model', 1)
        model=kwargs['dim_model']
        output=OrderedDict()
        debug_stream=[]
        idx=0
        for key,layer in model.layers.items():
            dict={}
            dict['layer']=layer
            dict['input']=layer.input
            dict['input_idx']=idx
            idx+=1
            dict['output'] = layer.output
            dict['output_idx'] = idx
            idx += 1
            output[key]=dict
            debug_stream.extend([layer.input,layer.output])
        debug_stream.extend([model.output.output,model.pred_Y,model.cost,model.error])
        debug_model = theano.function(inputs=model.inputs,
                                      outputs=debug_stream,
                                      on_unused_input='ignore')
        user_debug_model = theano.function(inputs=model.inputs,
                                      outputs=model.user_debug_stream,
                                      on_unused_input='ignore')

        train_X, valid_X, test_X, train_Y, valid_Y, test_Y = kwargs['data_stream']
        data = kwargs['prepare_data'](train_X, train_Y, range(self.debug_batch))
        data = tuple(data)
        debug_result=debug_model(*data)
        user_debug_result=user_debug_model(*data)
        kwargs['debug_result'].append(debug_result)
        kwargs['debug_result'].append(user_debug_result)

        self.logger("Debug model finished abort trainning",1,1)

        self.logger("Model Debug Info:\r\n\r\n",1)

        values=[]


        #★☆▲
        data=list(data)
        data=[d for d in data]
        self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)
        self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
        for d,inp in zip(data,model.inputs):
            self.logger('input:' +str(inp)+ ':'+str(np.array(d).shape), 3)
            self.logger(str(np.array(d))+'    id : %s'%len(values), 4)
            values.append(d)
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
        self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)
        for key, layer in output.items():
            self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)

            self.logger(layer['layer'].name + ':', 2, 1)
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
            self.logger('input:    %s'%layer['layer'].input + '  shape:'+str(debug_result[layer['input_idx']].shape), 3)
            self.logger('pprint:',3)
            self.logger(str(theano.pp(layer['layer'].input)) , 3)
            self.logger('value:',3)
            self.logger( str(debug_result[layer['input_idx']])+'    id : %s'%len(values), 4)
            values.append(debug_result[layer['input_idx']])

            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆',3)
            self.logger('params' + ':', 3)
            for param in layer['layer'].params:
                self.logger('%s'%param+' : '+str(param.get_value().shape),3)
                self.logger(str(param.get_value())+'    id : %s'%len(values),4)
                values.append(param.get_value())
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)

            self.logger('output:    %s' % layer['layer'].output + '  shape:' + str(debug_result[layer['output_idx']].shape), 3)
            self.logger('pprint:', 3)
            self.logger('▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲',3)
            self.logger(str(theano.pp(layer['layer'].output)) , 3)
            self.logger('▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲', 3)
            self.logger('value:', 3)
            self.logger(str(debug_result[layer['output_idx']])+'    id : %s'%len(values), 4)
            values.append(debug_result[layer['output_idx']])
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)

            self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)
            self.logger('output : ' +str(debug_result[-4].shape),2)
            self.logger(str(debug_result[-4])+'    id : %s'%len(values),3)
            values.append(debug_result[-4])
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
            self.logger('predict : ' + str(debug_result[-3].shape), 2)
            self.logger(str(debug_result[-3]) + '    id : %s' % len(values), 3)
            values.append(debug_result[-3])
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
            self.logger('cost : ' + str(debug_result[-2].shape), 2)
            self.logger(str(debug_result[-2]) + '    id : %s' % len(values), 3)
            values.append(debug_result[-2])
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
            self.logger('error : ' + str(debug_result[-1].shape), 2)
            self.logger(str(debug_result[-1]) + '    id : %s' % len(values), 3)
            values.append(debug_result[-1])
            self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)

        user_values=[]
        idx=0
        self.logger("\r\n\r\nUser Debug Info:\r\n\r\n", 1)

        self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)
        self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 2)
        for ud in model.user_debug_stream:

            self.logger('name : ' + str(ud)+'    '+str(ud.__name__), 2)
            self.logger(str(user_debug_result[idx]) + '    id : %s' % len(user_values), 3)
            user_values.append(user_debug_result[idx])
            idx+=1
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 2)

        self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)

        kwargs['stop'] = True

config=ex({})