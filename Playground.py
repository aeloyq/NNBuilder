# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 18:44:11 2016

@author: aeloyq
"""

import conf
import NNBuilder as nnb

if __name__ == '__main__':
    global conf, data_stream, model_stream,result_stream,vision_return
    conf = conf.get_conf_xor()
    data_stream = conf['data_pre'](conf)
    model_stream = nnb.Models.ModelBuilder.Model_Constructor(conf, data_stream)
    result_stream = nnb.Mainloop.Train.Train(conf, model_stream, data_stream)
    vision_return = nnb.Visions.Visualization.get_result(result_stream, model_stream)
