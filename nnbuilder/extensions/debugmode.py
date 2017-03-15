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

        #★☆▲


        for key, layer in output.items():
            self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)

            self.logger(layer['layer'].name + ':', 2, 1)
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)
            self.logger('input:    %s'%layer['layer'].input + '  shape:'+str(debug_result[layer['input_idx']].shape), 3)
            self.logger('pprint:',3)
            self.logger(str(theano.pp(layer['layer'].input)) , 3)
            self.logger('value:',3)
            self.logger( str(debug_result[layer['input_idx']]), 4)

            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆',3)
            self.logger('params' + ':', 3)
            for param in layer['layer'].params:
                self.logger('%s'%param+' : '+str(param.get_value().shape),3)
                self.logger(str(param.get_value()),4)
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)

            self.logger('output:    %s' % layer['layer'].output + '  shape:' + str(debug_result[layer['output_idx']].shape), 3)
            self.logger('pprint:', 3)
            self.logger('▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲',3)
            self.logger(str(theano.pp(layer['layer'].output)) , 3)
            self.logger('▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲', 3)
            self.logger('value:', 3)
            self.logger(str(debug_result[layer['output_idx']]), 4)
            self.logger('☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆☆', 3)

            self.logger('★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★', 2, 1)


        kwargs['stop'] = True

config=ex({})