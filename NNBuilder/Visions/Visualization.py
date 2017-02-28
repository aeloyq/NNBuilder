# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 23:24:31 2016

@author: aeloyq
"""

import matplotlib.pyplot as plt
def get_result(result_stream,model_stream):
    [total_epochs, errors, costs,debug_result]=result_stream
    datastream,train_model, valid_model, test_model, sample_model,debug_model, model, NNB_model,optimizer = model_stream
    x_axis = range(total_epochs)
    plt.plot(x_axis, errors)
    plt.plot(x_axis, costs)
    plt.show()